import argparse
import os
import random
from datetime import datetime
import time

import d4rl
import gymnasium as gym
import minari
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from data import D4RLTrajectoryDataset
from trainer import ReinFormerTrainer
from eval import Reinformer_eval


def reward_scale_for_env(env_name: str) -> float:
    if env_name in ["hopper", "walker2d"]:
        return 1000.0
    if env_name in ["halfcheetah"]:
        return 5000.0
    if env_name in ["maze2d", "kitchen"]:
        return 100.0
    if env_name in ["pen", "door", "hammer", "relocate"]:
        return 10000.0
    if env_name in ["antmaze"]:
        return 1.0
    return 1.0


def discount_cumsum_np(x, gamma=1.0):
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def experiment(variant):
    # seeding
    seed = variant["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = variant["env"]
    dataset = variant["dataset"]
    
    if dataset == "complete":
        variant["batch_size"] = 16

    # Build a D4RL-style ID for locating the offline dataset on disk.
    # Most tasks follow the original D4RL versioning, but some Minari-only
    # datasets (e.g. hopper-simple-v0) are handled specially.
    if env == "kitchen":
        d4rl_env = f"{env}-{dataset}-v0"
    elif env == "hopper" and dataset == "simple":
        # Minari-only dataset; we save trajectories under this pseudo-D4RL ID.
        d4rl_env = "hopper-simple-v0"
    elif env in ["pen", "door", "hammer", "relocate", "maze2d"]:
        d4rl_env = f"{env}-{dataset}-v1"
    elif env in ["halfcheetah", "hopper", "walker2d", "antmaze"]:
        d4rl_env = f"{env}-{dataset}-v2"
    if env in ["kitchen", "maze2d"]:
        variant["num_eval_ep"] = 100
    if env == "antmaze":
        variant["num_eval_ep"] = min(variant["num_eval_ep"], 10)
        variant["max_eval_ep_len"] = min(variant["max_eval_ep_len"], 300)
    if env == "hopper":
        if dataset == "medium" or dataset == "meidum-replay":
            variant["batch_size"] = 256
    
    dataset_path = os.path.join(variant["dataset_dir"], f"{d4rl_env}.pkl")
    device = torch.device(variant["device"])

    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
    # Per-run plot directory to avoid overwriting plots across runs.
    # Prefer an explicit directory provided via environment variable
    # (e.g., set by Slurm scripts to mirror log structure), otherwise
    # fall back to a job-based directory under RF/plots.
    plot_run_dir_env = os.environ.get("PLOT_RUN_DIR")
    if plot_run_dir_env:
        plot_run_dir = plot_run_dir_env
    else:
        slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get(
            "SLURM_JOB_ID"
        )
        if slurm_job_id:
            plot_run_dir = os.path.join(
                "plots", f"{d4rl_env}_job{slurm_job_id}_{start_time_str}"
            )
        else:
            plot_run_dir = os.path.join("plots", f"{d4rl_env}_{start_time_str}")

    # Per-run plot directory to avoid overwriting plots and to encode settings.
    plot_run_dir_env = os.environ.get("PLOT_RUN_DIR")
    if plot_run_dir_env:
        plot_run_dir = plot_run_dir_env
    else:
        # Use a simple convention: offline vs online, optional beta/adv, and job id.
        slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get(
            "SLURM_JOB_ID"
        )
        if variant.get("online_training", False):
            beta_rl = variant.get("beta_rl", 0.0)
            adv_scale = variant.get("adv_scale", 1.0)
            subdir = os.path.join(
                "online",
                f"beta_{beta_rl}_adv_{adv_scale}",
            )
        else:
            subdir = "offline"

        if slurm_job_id:
            plot_run_dir = os.path.join("plots", subdir, f"job_{slurm_job_id}")
        else:
            plot_run_dir = os.path.join("plots", subdir, start_time_str)

    print("=" * 60)
    print("start time: " + start_time_str)
    print("=" * 60)

    traj_dataset = D4RLTrajectoryDataset(
        env, dataset_path, variant["context_len"], device
    )

    traj_data_loader = DataLoader(
        traj_dataset,
        batch_size=variant["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    data_iter = iter(traj_data_loader)

    state_mean, state_std = traj_dataset.get_state_stats()

    # Infer dimensions from offline dataset to avoid depending on env creation.
    sample_traj = traj_dataset.trajectories[0]
    state_dim = sample_traj["observations"].shape[1]
    act_dim = sample_traj["actions"].shape[1]
    reward_scale = reward_scale_for_env(env)

    def preprocess_reward(r):
        if env == "antmaze":
            return r * 100 + 1
        return r

    def compute_returns_to_go_from_rewards(rewards_np):
        return discount_cumsum_np(rewards_np, gamma=variant.get("rtg_gamma", 1.0)) / reward_scale

    def collect_online_trajectory(model, env_for_rollout, use_mean_action=False):
        eval_batch_size = 1
        states = []
        actions = []
        rewards_list = []

        # infer dimensions from dataset statistics / env action space
        state_dim_local = state_mean.shape[0]
        act_space = env_for_rollout.action_space
        if getattr(act_space, "shape", None) is None:
            raise ValueError("Unsupported action space without shape attribute.")
        act_dim_local = act_space.shape[0]

        state_mean_t = torch.from_numpy(state_mean).to(device)
        state_std_t = torch.from_numpy(state_std).to(device)

        timesteps_all = torch.arange(
            start=0, end=variant["max_eval_ep_len"], step=1
        ).repeat(eval_batch_size, 1).to(device)

        model.eval()
        with torch.no_grad():
            obs, _ = env_for_rollout.reset()

            def _extract_obs(o):
                if isinstance(o, dict):
                    if "observation" in o:
                        return o["observation"]
                    raise ValueError(
                        f"Dict observation missing 'observation' key: {list(o.keys())}"
                    )
                return o

            running_state = _extract_obs(obs)

            actions_buf = torch.zeros(
                (eval_batch_size, variant["max_eval_ep_len"], act_dim_local),
                dtype=torch.float32,
                device=device,
            )
            states_buf = torch.zeros(
                (eval_batch_size, variant["max_eval_ep_len"], state_dim_local),
                dtype=torch.float32,
                device=device,
            )
            returns_to_go_buf = torch.zeros(
                (eval_batch_size, variant["max_eval_ep_len"], 1),
                dtype=torch.float32,
                device=device,
            )

            for t in range(variant["max_eval_ep_len"]):
                states_buf[0, t] = torch.from_numpy(running_state).to(device)
                states_buf[0, t] = (states_buf[0, t] - state_mean_t) / state_std_t

                if t < variant["context_len"]:
                    rtg_preds, act_dist_preds, _ = model.forward(
                        timesteps_all[:, : variant["context_len"]],
                        states_buf[:, : variant["context_len"]],
                        actions_buf[:, : variant["context_len"]],
                        returns_to_go_buf[:, : variant["context_len"]],
                    )
                    rtg = rtg_preds[0, t].detach()
                    act_dist = act_dist_preds
                    act = (
                        act_dist.mean.reshape(eval_batch_size, -1, act_dim_local)[
                            0, t
                        ].detach()
                        if use_mean_action
                        else act_dist.rsample().reshape(
                            eval_batch_size, -1, act_dim_local
                        )[0, t].detach()
                    )
                else:
                    rtg_preds, act_dist_preds, _ = model.forward(
                        timesteps_all[:, t - variant["context_len"] + 1 : t + 1],
                        states_buf[:, t - variant["context_len"] + 1 : t + 1],
                        actions_buf[:, t - variant["context_len"] + 1 : t + 1],
                        returns_to_go_buf[:, t - variant["context_len"] + 1 : t + 1],
                    )
                    rtg = rtg_preds[0, -1].detach()
                    act_dist = act_dist_preds
                    act = (
                        act_dist.mean.reshape(eval_batch_size, -1, act_dim_local)[
                            0, -1
                        ].detach()
                        if use_mean_action
                        else act_dist.rsample().reshape(
                            eval_batch_size, -1, act_dim_local
                        )[0, -1].detach()
                    )

                returns_to_go_buf[0, t] = rtg
                actions_buf[0, t] = act

                (
                    obs,
                    running_reward,
                    terminated,
                    truncated,
                    _,
                ) = env_for_rollout.step(act.cpu().numpy())
                running_state = _extract_obs(obs)

                rewards_list.append(preprocess_reward(running_reward))
                states.append(states_buf[0, t].cpu().numpy())
                actions.append(act.cpu().numpy())

                done = bool(terminated or truncated)
                if done:
                    break

        traj_len = len(rewards_list)
        returns_to_go_np = compute_returns_to_go_from_rewards(
            np.array(rewards_list, dtype=np.float32)
        )
        return {
            "observations": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards_list, dtype=np.float32),
            "returns_to_go": returns_to_go_np.astype(np.float32),
            "traj_len": traj_len,
        }

    def sample_online_batch(replay_buffer, batch_size, context_len):
        batch_timesteps, batch_states, batch_actions = [], [], []
        batch_returns_to_go, batch_rewards, batch_masks = [], [], []

        for _ in range(batch_size):
            traj = random.choice(replay_buffer)
            traj_len = traj["observations"].shape[0]

            if traj_len >= context_len:
                si = random.randint(0, traj_len - context_len)
                states_slice = traj["observations"][si : si + context_len]
                actions_slice = traj["actions"][si : si + context_len]
                returns_slice = traj["returns_to_go"][si : si + context_len]
                rewards_slice = traj["rewards"][si : si + context_len]
                timesteps = np.arange(si, si + context_len)
                mask = np.ones(context_len, dtype=np.float32)
            else:
                padding_len = context_len - traj_len
                states_slice = np.concatenate(
                    [
                        traj["observations"],
                        np.zeros((padding_len, state_dim), dtype=np.float32),
                    ],
                    axis=0,
                )
                actions_slice = np.concatenate(
                    [
                        traj["actions"],
                        np.zeros((padding_len, act_dim), dtype=np.float32),
                    ],
                    axis=0,
                )
                returns_slice = np.concatenate(
                    [
                        traj["returns_to_go"],
                        np.zeros((padding_len,), dtype=np.float32),
                    ],
                    axis=0,
                )
                rewards_slice = np.concatenate(
                    [
                        traj["rewards"],
                        np.zeros((padding_len,), dtype=np.float32),
                    ],
                    axis=0,
                )
                timesteps = np.arange(0, context_len)
                mask = np.concatenate(
                    [
                        np.ones(traj_len, dtype=np.float32),
                        np.zeros(padding_len, dtype=np.float32),
                    ],
                    axis=0,
                )

            batch_timesteps.append(timesteps)
            batch_states.append(states_slice)
            batch_actions.append(actions_slice)
            batch_returns_to_go.append(returns_slice)
            batch_rewards.append(rewards_slice)
            batch_masks.append(mask)

        timesteps_t = torch.from_numpy(np.stack(batch_timesteps)).long().to(device)
        states_t = torch.from_numpy(np.stack(batch_states)).float().to(device)
        actions_t = torch.from_numpy(np.stack(batch_actions)).float().to(device)
        returns_t = torch.from_numpy(np.stack(batch_returns_to_go)).float().to(device)
        rewards_t = torch.from_numpy(np.stack(batch_rewards)).float().to(device)
        masks_t = torch.from_numpy(np.stack(batch_masks)).long().to(device)

        return (
            timesteps_t,
            states_t,
            actions_t,
            returns_t,
            rewards_t,
            masks_t,
        )

    # Try to create an evaluation environment if available.
    eval_env = None
    eval_env_id = None
    eval_dataset_for_norm = None
    use_minari_norm = False

    # For some tasks, prefer the Minari-provided evaluation environment.
    if env == "antmaze" and dataset == "medium-diverse":
        try:
            minari_id = "D4RL/antmaze/medium-diverse-v1"
            ds = minari.load_dataset(minari_id)
            eval_dataset_for_norm = ds
            eval_env = ds.recover_environment(
                eval_env=True, max_episode_steps=variant["max_eval_ep_len"]
            )
            eval_env.reset(seed=seed)
            env_spec = ds._eval_env_spec or ds.spec.env_spec
            eval_env_id = env_spec.id if env_spec is not None else d4rl_env
            use_minari_norm = True
            print(f"Using Minari eval environment: {eval_env_id}")
        except Exception as e:
            print(
                f"Warning: could not recover AntMaze evaluation env from Minari. "
                f"Falling back to D4RL id '{d4rl_env}' if available. Error: {e}"
            )
    elif env == "hopper" and dataset in ["medium", "simple"]:
        try:
            if dataset == "medium":
                minari_id = "mujoco/hopper/medium-v0"
            else:
                # hopper-simple-v0 Minari dataset
                minari_id = "mujoco/hopper/simple-v0"
            ds = minari.load_dataset(minari_id)
            eval_dataset_for_norm = ds
            eval_env = ds.recover_environment(
                eval_env=True, max_episode_steps=variant["max_eval_ep_len"]
            )
            eval_env.reset(seed=seed)
            env_spec = ds._eval_env_spec or ds.spec.env_spec
            eval_env_id = env_spec.id if env_spec is not None else d4rl_env
            # For hopper-medium-v0, Minari currently does not provide
            # ref_min_score/ref_max_score, so we do not use Minari
            # normalization there. For hopper-simple-v0 we try Minari
            # normalization (if available) via use_minari_norm.
            if dataset == "simple":
                use_minari_norm = True
            else:
                use_minari_norm = False
            print(f"Using Minari eval environment: {eval_env_id}")
        except Exception as e:
            print(
                f"Warning: could not recover Hopper evaluation env from Minari. "
                f"Falling back to D4RL id '{d4rl_env}' if available. Error: {e}"
            )
    elif env == "walker2d" and dataset == "medium":
        try:
            minari_id = "mujoco/walker2d/medium-v0"
            ds = minari.load_dataset(minari_id)
            eval_dataset_for_norm = ds
            eval_env = ds.recover_environment(
                eval_env=True, max_episode_steps=variant["max_eval_ep_len"]
            )
            eval_env.reset(seed=seed)
            env_spec = ds._eval_env_spec or ds.spec.env_spec
            eval_env_id = env_spec.id if env_spec is not None else d4rl_env
            # Walker2d-medium-v0 has D4RL reference scores; we prefer
            # Minari normalization when available.
            use_minari_norm = True
            print(f"Using Minari eval environment: {eval_env_id}")
        except Exception as e:
            print(
                f"Warning: could not recover Walker2d evaluation env from Minari. "
                f"Falling back to D4RL id '{d4rl_env}' if available. Error: {e}"
            )
    elif env == "maze2d" and dataset == "medium":
        try:
            minari_id = "D4RL/pointmaze/medium-v2"
            ds = minari.load_dataset(minari_id)
            eval_dataset_for_norm = ds
            eval_env = ds.recover_environment(
                eval_env=True, max_episode_steps=variant["max_eval_ep_len"]
            )
            eval_env.reset(seed=seed)
            env_spec = ds._eval_env_spec or ds.spec.env_spec
            eval_env_id = env_spec.id if env_spec is not None else d4rl_env
            use_minari_norm = True
            print(f"Using Minari eval environment: {eval_env_id}")
        except Exception as e:
            print(
                f"Warning: could not recover Maze2d evaluation env from Minari. "
                f"Falling back to D4RL id '{d4rl_env}' if available. Error: {e}"
            )
    elif env == "kitchen" and dataset == "mixed":
        try:
            minari_id = "D4RL/kitchen/mixed-v2"
            ds = minari.load_dataset(minari_id)
            eval_dataset_for_norm = ds
            eval_env = ds.recover_environment(
                eval_env=True, max_episode_steps=variant["max_eval_ep_len"]
            )
            eval_env.reset(seed=seed)
            env_spec = ds._eval_env_spec or ds.spec.env_spec
            eval_env_id = env_spec.id if env_spec is not None else d4rl_env
            use_minari_norm = True
            print(f"Using Minari eval environment: {eval_env_id}")
        except Exception as e:
            print(
                f"Warning: could not recover Kitchen evaluation env from Minari. "
                f"Falling back to D4RL id '{d4rl_env}' if available. Error: {e}"
            )

    # Fallback: try D4RL-registered env (if available).
    if eval_env is None:
        try:
            eval_env_id = d4rl_env
            eval_env = gym.make(d4rl_env)
            eval_env.reset(seed=seed)
            use_minari_norm = False
            print(f"Using D4RL env for evaluation: {d4rl_env}")
        except Exception as e:
            print(
                f"Warning: could not create evaluation env '{d4rl_env}'. "
                f"Online evaluation will be skipped. Error: {e}"
            )

    model_type = variant["model_type"]

    evaluator = None
    if model_type == "reinformer":
        Trainer = ReinFormerTrainer(
            state_dim=state_dim,
            act_dim=act_dim,
            device=device,
            variant=variant
        )

        # If a pretrained checkpoint is provided, load it before any evaluation
        # or training so that the initial eval reflects the checkpoint.
        if variant.get("pretrained_model"):
            try:
                state_dict = torch.load(variant["pretrained_model"], map_location=device)
                Trainer.model.load_state_dict(state_dict)
                print(f"Loaded pretrained model from {variant['pretrained_model']}")
            except Exception as e:
                print(f"Warning: could not load pretrained model: {e}")

        if eval_env is not None:
            def evaluator(model):
                return_mean, _, _, _ = Reinformer_eval(
                    model=model,
                    device=device,
                    context_len=variant["context_len"],
                    env=eval_env,
                    state_mean=state_mean,
                    state_std=state_std,
                    num_eval_ep=variant["num_eval_ep"],
                    max_test_ep_len=variant["max_eval_ep_len"],
                )
                raw_return = float(return_mean)

                normalized_score = None

                # Prefer Minari normalization if reference scores are available.
                if use_minari_norm and eval_dataset_for_norm is not None:
                    try:
                        norm_score = minari.get_normalized_score(
                            eval_dataset_for_norm, np.array([raw_return])
                        )[0]
                        normalized_score = norm_score * 100.0
                    except ValueError:
                        # Dataset does not have reference scores; for
                        # hopper-simple, fall back to normalizing by the
                        # min/max returns observed in the offline dataset.
                        if env == "hopper" and dataset == "simple":
                            try:
                                ret_max, _, _, ret_min = traj_dataset.get_return_stats()
                                if ret_max > ret_min:
                                    norm = (raw_return - ret_min) / (ret_max - ret_min)
                                    normalized_score = norm * 100.0
                            except Exception:
                                pass
                # For dense Mujoco tasks like hopper and walker2d, compute
                # D4RL-style normalized score manually using REF_MIN/REF_MAX
                # from d4rl for the classic D4RL medium datasets.
                if (
                    normalized_score is None
                    and env in ["hopper", "walker2d"]
                    and dataset == "medium"
                ):
                    try:
                        from d4rl import infos as d4rl_infos

                        key_medium = f"{env}-medium-v0"
                        key_random = f"{env}-random-v0"
                        ref_min = d4rl_infos.REF_MIN_SCORE.get(
                            key_medium,
                            d4rl_infos.REF_MIN_SCORE[key_random],
                        )
                        ref_max = d4rl_infos.REF_MAX_SCORE.get(
                            key_medium,
                            d4rl_infos.REF_MAX_SCORE[key_random],
                        )
                        norm = (raw_return - ref_min) / (ref_max - ref_min)
                        normalized_score = norm * 100.0
                    except Exception:
                        pass
                # Next, try D4RL-style env normalization if available.
                if normalized_score is None and hasattr(eval_env, "get_normalized_score"):
                    normalized_score = eval_env.get_normalized_score(raw_return) * 100.0
                # Fallback: if normalization still not available, just use raw return.
                if normalized_score is None:
                    normalized_score = raw_return

                return raw_return, normalized_score

    max_train_iters = variant["max_train_iters"]
    num_updates_per_iter = variant["num_updates_per_iter"]
    normalized_d4rl_score_list = []
    raw_return_list = []

    # Global gradient-update counter (for throttled logging).

    # Per-iteration training diagnostics for plotting.
    training_loss_list = []
    rtg_loss_list = []
    action_loss_list = []
    temperature_loss_list = []
    entropy_list = []
    temperature_list = []
    grad_norm_list = []
    rl_loss_list = []
    policy_loss_list = []
    critic_loss_list = []

    # Global gradient-update counter (for throttled logging).

    # Per-iteration training diagnostics for plotting.
    training_loss_list = []
    rtg_loss_list = []
    action_loss_list = []
    temperature_loss_list = []
    entropy_list = []
    temperature_list = []
    grad_norm_list = []
    rl_loss_list = []

    # Training diagnostics (per-iteration and per-update) for plotting/logging.
    training_loss_list = []
    rtg_loss_list = []
    action_loss_list = []
    temperature_loss_list = []
    rl_loss_list = []
    entropy_list = []
    temperature_list = []
    grad_norm_list = []

    step_loss_list = []
    step_rtg_loss_list = []
    step_action_loss_list = []
    step_temperature_loss_list = []
    step_rl_loss_list = []
    step_policy_loss_list = []
    step_critic_loss_list = []
    step_entropy_list = []
    step_temperature_value_list = []
    step_grad_norm_list = []

    online_training = variant.get("online_training", False)
    reinforce_online = variant.get("reinforce_online", False)

    # Always run an initial evaluation (if possible) before any training updates.
    if evaluator is not None:
        print("Running initial evaluation before training...")
        init_raw_return, init_score = evaluator(model=Trainer.model)
        raw_return_list.append(init_raw_return)
        normalized_d4rl_score_list.append(init_score)
        if args.use_wandb:
            wandb.log(
                data={
                    "evaluation/score": init_score,
                    "evaluation/score_normalized": init_score,
                    "evaluation/score_raw": init_raw_return,
                    "evaluation/iteration": 0,
                }
            )

    if not online_training:

        # Offline training loop with diagnostics.
        for itr in tqdm(range(1, max_train_iters + 1), desc="Training iterations"):
            t1 = time.time()
            total_loss_sum = 0.0
            rtg_loss_sum = 0.0
            action_loss_sum = 0.0
            temperature_loss_sum = 0.0
            rl_loss_sum = 0.0
            policy_loss_sum = 0.0
            critic_loss_sum = 0.0
            entropy_sum = 0.0
            temperature_sum = 0.0
            grad_norm_sum = 0.0
            num_updates = 0

            for _ in range(num_updates_per_iter):
                try:
                    (
                        timesteps,
                        states,
                        actions,
                        returns_to_go,
                        rewards,
                        traj_mask,
                    ) = next(data_iter)
                except StopIteration:
                    data_iter = iter(traj_data_loader)
                    (
                        timesteps,
                        states,
                        actions,
                        returns_to_go,
                        rewards,
                        traj_mask,
                    ) = next(data_iter)

                loss, diag = Trainer.train_step(
                    timesteps=timesteps,
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go,
                    rewards=rewards,
                    traj_mask=traj_mask,
                )

                # Accumulate diagnostics for this iteration.
                total_loss_sum += loss
                rtg_loss_sum += diag["returns_to_go_loss"]
                action_loss_sum += diag["action_loss"]
                temperature_loss_sum += diag["temperature_loss"]
                rl_loss_sum += diag["rl_loss"]
                policy_loss_sum += diag.get("policy_loss", diag["rl_loss"])
                critic_loss_sum += diag.get("critic_loss", diag["returns_to_go_loss"])
                entropy_sum += diag["entropy"]
                temperature_sum += diag["temperature"]
                grad_norm_sum += diag["grad_norm"]
                num_updates += 1

                # Fine-grained per-update diagnostics.
                step_loss_list.append(loss)
                step_rtg_loss_list.append(diag["returns_to_go_loss"])
                step_action_loss_list.append(diag["action_loss"])
                step_temperature_loss_list.append(diag["temperature_loss"])
                step_rl_loss_list.append(diag["rl_loss"])
                step_policy_loss_list.append(diag.get("policy_loss", diag["rl_loss"]))
                step_critic_loss_list.append(diag.get("critic_loss", diag["returns_to_go_loss"]))
                step_entropy_list.append(diag["entropy"])
                step_temperature_value_list.append(diag["temperature"])
                step_grad_norm_list.append(diag["grad_norm"])

                if args.use_wandb:
                    wandb.log(
                        data={
                            "training/loss": loss,
                            "training/returns_to_go_loss": diag["returns_to_go_loss"],
                            "training/action_loss": diag["action_loss"],
                            "training/temperature_loss": diag["temperature_loss"],
                            "training/rl_loss": diag["rl_loss"],
                            "training/policy_loss": diag.get("policy_loss", diag["rl_loss"]),
                            "training/critic_loss": diag.get("critic_loss", diag["returns_to_go_loss"]),
                            "training/entropy": diag["entropy"],
                            "training/temperature": diag["temperature"],
                            "training/grad_norm": diag["grad_norm"],
                        }
                    )

            # Per-iteration averages (used for coarse plots).
            if num_updates > 0:
                training_loss_list.append(total_loss_sum / num_updates)
                rtg_loss_list.append(rtg_loss_sum / num_updates)
                action_loss_list.append(action_loss_sum / num_updates)
                temperature_loss_list.append(temperature_loss_sum / num_updates)
                rl_loss_list.append(rl_loss_sum / num_updates)
                policy_loss_list.append(policy_loss_sum / num_updates)
                critic_loss_list.append(critic_loss_sum / num_updates)
                entropy_list.append(entropy_sum / num_updates)
                temperature_list.append(temperature_sum / num_updates)
                grad_norm_list.append(grad_norm_sum / num_updates)

            t2 = time.time()
            if evaluator is not None:
                raw_return, normalized_d4rl_score = evaluator(model=Trainer.model)
                t3 = time.time()
                raw_return_list.append(raw_return)
                normalized_d4rl_score_list.append(normalized_d4rl_score)
                if args.use_wandb:
                    wandb.log(
                        data={
                            "training/time": t2 - t1,
                            "evaluation/score": normalized_d4rl_score,
                            "evaluation/score_normalized": normalized_d4rl_score,
                            "evaluation/score_raw": raw_return,
                            "evaluation/time": t3 - t2,
                        }
                    )

            # Simple textual indicator that offline training is progressing.
            print(
                f"Iteration {itr}/{max_train_iters} - "
                f"last training loss: {loss:.4f}"
                + (
                    f", last eval raw return: {raw_return:.2f}, "
                    f"last eval normalized score: {normalized_d4rl_score:.2f}"
                    if evaluator is not None
                    else ""
                )
            )
    else:
        # Online finetuning loop with advantage-weighted RL term or REINFORCE.
        online_buffer_size = variant.get("online_buffer_size")
        if online_buffer_size is None:
            online_buffer_size = len(traj_dataset)
        beta_rl = variant.get("beta_rl", 0.0)
        adv_scale = variant.get("adv_scale", 1.0)
        num_online_rollouts_per_iter = int(variant.get("num_online_rollouts_per_iter", 1))
        use_baseline = bool(variant.get("baseline", False))
        expectile_tau = float(variant.get("tau", 0.99))
        critic_coef = float(variant.get("critic_coef", 1.0))
        no_buffer = variant.get("no_buffer", False)

        # Initialize buffer. If reinforce_online && no_buffer, skip seeding and
        # avoid reusing past rollouts. If reinforce_online but buffer reuse is
        # allowed, start empty. Otherwise seed with top offline.
        if reinforce_online and no_buffer:
            online_replay_buffer = []
        elif reinforce_online:
            online_replay_buffer = []
        else:
            offline_returns = [
                float(np.sum(traj["rewards"])) for traj in traj_dataset.trajectories
            ]
            sorted_indices = np.argsort(offline_returns)[::-1]
            online_replay_buffer = [
                traj_dataset.trajectories[i]
                for i in sorted_indices[:online_buffer_size]
            ]

        try:
            rollout_env = gym.make(d4rl_env)
            rollout_env.reset(seed=seed + 1 if seed is not None else None)
        except Exception:
            rollout_env = eval_env

        for itr in tqdm(range(1, max_train_iters + 1), desc="Online iterations"):
            # 1) Collect rollouts with stochastic policy (rsample).
            new_rollouts = []
            for _ in range(num_online_rollouts_per_iter):
                new_traj = collect_online_trajectory(
                    model=Trainer.model,
                    env_for_rollout=rollout_env,
                    use_mean_action=False,
                )
                new_rollouts.append(new_traj)
                if not (reinforce_online and no_buffer):
                    online_replay_buffer.append(new_traj)
                    if len(online_replay_buffer) > online_buffer_size:
                        online_replay_buffer.pop(0)

            # 2) Train for num_updates_per_iter minibatches from buffer.
            t1 = time.time()
            total_loss_sum = 0.0
            rtg_loss_sum = 0.0
            action_loss_sum = 0.0
            temperature_loss_sum = 0.0
            entropy_sum = 0.0
            temperature_sum = 0.0
            grad_norm_sum = 0.0
            rl_loss_sum = 0.0
            policy_loss_sum = 0.0
            critic_loss_sum = 0.0
            num_updates = 0

            for _ in range(num_updates_per_iter):
                if reinforce_online and no_buffer:
                    # Train only on freshly collected rollouts; batch size equals rollouts collected.
                    (
                        timesteps,
                        states,
                        actions,
                        returns_to_go,
                        rewards,
                        traj_mask,
                    ) = sample_online_batch(
                        new_rollouts,
                        batch_size=len(new_rollouts),
                        context_len=variant["context_len"],
                    )
                    loss, diag = Trainer.reinforce_step(
                        timesteps=timesteps,
                        states=states,
                        actions=actions,
                        returns_to_go=returns_to_go,
                        traj_mask=traj_mask,
                        use_baseline=use_baseline,
                        tau=expectile_tau,
                        critic_coef=critic_coef,
                    )
                else:
                    (
                        timesteps,
                        states,
                        actions,
                        returns_to_go,
                        rewards,
                        traj_mask,
                    ) = sample_online_batch(
                        online_replay_buffer,
                        batch_size=variant["batch_size"],
                        context_len=variant["context_len"],
                    )
                    if reinforce_online:
                        loss, diag = Trainer.reinforce_step(
                            timesteps=timesteps,
                            states=states,
                            actions=actions,
                            returns_to_go=returns_to_go,
                            traj_mask=traj_mask,
                            use_baseline=use_baseline,
                            tau=expectile_tau,
                            critic_coef=critic_coef,
                        )
                    else:
                        loss, diag = Trainer.train_step(
                            timesteps=timesteps,
                            states=states,
                            actions=actions,
                            returns_to_go=returns_to_go,
                            rewards=rewards,
                            traj_mask=traj_mask,
                            beta_rl=beta_rl,
                            adv_scale=adv_scale,
                        )
                # Fine-grained per-update diagnostics for online phase as well.
                step_loss_list.append(loss)
                step_rtg_loss_list.append(diag["returns_to_go_loss"])
                step_action_loss_list.append(diag["action_loss"])
                step_temperature_loss_list.append(diag["temperature_loss"])
                step_rl_loss_list.append(diag["rl_loss"])
                step_policy_loss_list.append(diag.get("policy_loss", diag["rl_loss"]))
                step_critic_loss_list.append(diag.get("critic_loss", diag["returns_to_go_loss"]))
                step_entropy_list.append(diag["entropy"])
                step_temperature_value_list.append(diag["temperature"])
                step_grad_norm_list.append(diag["grad_norm"])

                if args.use_wandb:
                    wandb.log(
                        data={
                            "training/loss": loss,
                            "training/returns_to_go_loss": diag["returns_to_go_loss"],
                            "training/action_loss": diag["action_loss"],
                            "training/temperature_loss": diag["temperature_loss"],
                            "training/rl_loss": diag["rl_loss"],
                            "training/policy_loss": diag.get("policy_loss", diag["rl_loss"]),
                            "training/critic_loss": diag.get("critic_loss", diag["returns_to_go_loss"]),
                            "training/entropy": diag["entropy"],
                            "training/temperature": diag["temperature"],
                            "training/grad_norm": diag["grad_norm"],
                        }
                    )

                # accumulate per-iteration sums (online)
                total_loss_sum += loss
                rtg_loss_sum += diag["returns_to_go_loss"]
                action_loss_sum += diag["action_loss"]
                temperature_loss_sum += diag["temperature_loss"]
                entropy_sum += diag["entropy"]
                temperature_sum += diag["temperature"]
                grad_norm_sum += diag["grad_norm"]
                rl_loss_sum += diag["rl_loss"]
                policy_loss_sum += diag.get("policy_loss", diag["rl_loss"])
                critic_loss_sum += diag.get("critic_loss", diag["returns_to_go_loss"])
                num_updates += 1

            t2 = time.time()

            # Per-iteration averages for plotting (online).
            if num_updates > 0:
                training_loss_list.append(total_loss_sum / num_updates)
                rtg_loss_list.append(rtg_loss_sum / num_updates)
                action_loss_list.append(action_loss_sum / num_updates)
                temperature_loss_list.append(temperature_loss_sum / num_updates)
                entropy_list.append(entropy_sum / num_updates)
                temperature_list.append(temperature_sum / num_updates)
                grad_norm_list.append(grad_norm_sum / num_updates)
                rl_loss_list.append(rl_loss_sum / num_updates)
                policy_loss_list.append(policy_loss_sum / num_updates)
                critic_loss_list.append(critic_loss_sum / num_updates)

            # 3) Evaluate with deterministic actions.
            if evaluator is not None:
                raw_return, normalized_d4rl_score = evaluator(model=Trainer.model)
                t3 = time.time()
                raw_return_list.append(raw_return)
                normalized_d4rl_score_list.append(normalized_d4rl_score)
                if args.use_wandb:
                    wandb.log(
                        data={
                            "training/time": t2 - t1,
                            "evaluation/score": normalized_d4rl_score,
                            "evaluation/score_normalized": normalized_d4rl_score,
                            "evaluation/score_raw": raw_return,
                            "evaluation/time": t3 - t2,
                        }
                    )

            print(
                f"[Online] Iteration {itr}/{max_train_iters} - "
                f"last training loss: {loss:.4f}"
                + (
                    f", last eval raw return: {raw_return:.2f}, "
                    f"last eval normalized score: {normalized_d4rl_score:.2f}"
                    if evaluator is not None
                    else ""
                )
            )

    # Log final evaluation summary to wandb if enabled.
    if normalized_d4rl_score_list and args.use_wandb:
        wandb.log(
            data={
                "evaluation/max_score": max(normalized_d4rl_score_list),
                "evaluation/last_score": normalized_d4rl_score_list[-1],
            }
        )
    # Print and plot evaluation and training diagnostics.
    if normalized_d4rl_score_list:
        print("Evaluation scores over iterations:", normalized_d4rl_score_list)

        if raw_return_list:
            print("Raw returns over iterations:", raw_return_list)

        # Plot evaluation curves and training diagnostics if matplotlib is available.
        if plt is not None:
            os.makedirs("plots", exist_ok=True)
            # Use start_time_str to avoid overwriting previous runs.
            plot_path_norm = os.path.join(
                plot_run_dir, f"{d4rl_env}_eval_{start_time_str}.png"
            )
            plt.figure()
            plt.plot(normalized_d4rl_score_list, marker="o", label="raw")
            # Smoothed (moving-average) trajectory for readability.
            if len(normalized_d4rl_score_list) > 2:
                smooth_window = max(1, len(normalized_d4rl_score_list) // 10)
                weights = np.ones(smooth_window) / smooth_window
                smoothed_scores = np.convolve(
                    normalized_d4rl_score_list, weights, mode="valid"
                )
                x_smoothed = np.arange(smooth_window, smooth_window + len(smoothed_scores))
                plt.plot(x_smoothed, smoothed_scores, label=f"smoothed (w={smooth_window})")
            plt.legend()
            plt.xlabel("Training iteration")
            plt.ylabel("Normalized score")
            plt.title(f"Online evaluation - {eval_env_id or d4rl_env}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path_norm)
            plt.close()

            if raw_return_list:
                plot_path_raw = os.path.join(
                    plot_run_dir, f"{d4rl_env}_eval_raw_return_{start_time_str}.png"
                )
                plt.figure()
                plt.plot(raw_return_list, marker="o")
                plt.xlabel("Training iteration")
                plt.ylabel("Average return per episode")
                plt.title(
                    f"Online evaluation (raw returns) - {eval_env_id or d4rl_env}"
                )
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plot_path_raw)
                plt.close()
                print(f"Saved raw return evaluation plot to {plot_path_raw}")

            print(f"Saved normalized evaluation plot to {plot_path_norm}")

            # Training loss/components plot.
            if training_loss_list:
                iter_steps = list(range(1, len(training_loss_list) + 1))
                train_loss_path = os.path.join(
                    plot_run_dir, f"{d4rl_env}_training_losses_{start_time_str}.png"
                )
                plt.figure()
                plt.plot(iter_steps, training_loss_list, label="total_loss")
                plt.plot(iter_steps, rtg_loss_list, label="rtg_loss")
                plt.plot(iter_steps, action_loss_list, label="action_loss")
                plt.plot(iter_steps, temperature_loss_list, label="temperature_loss")
                plt.plot(iter_steps, rl_loss_list, label="rl_loss")
                if policy_loss_list:
                    plt.plot(iter_steps, policy_loss_list, label="policy_loss")
                if critic_loss_list:
                    plt.plot(iter_steps, critic_loss_list, label="critic_loss")
                plt.xlabel("Training iteration")
                plt.ylabel("Loss")
                plt.title(f"Training losses (per-iteration) - {eval_env_id or d4rl_env}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(train_loss_path)
                plt.close()
                print(f"Saved training loss plot to {train_loss_path}")

            # Policy stats plot (entropy, temperature, grad_norm).
            if entropy_list or temperature_list or grad_norm_list:
                iter_steps = list(range(1, len(entropy_list) + 1))
                stats_path = os.path.join(
                    plot_run_dir, f"{d4rl_env}_training_stats_{start_time_str}.png"
                )
                plt.figure()
                if entropy_list:
                    plt.plot(iter_steps, entropy_list, label="entropy")
                if temperature_list:
                    plt.plot(iter_steps, temperature_list, label="temperature")
                if grad_norm_list:
                    plt.plot(iter_steps, grad_norm_list, label="grad_norm")
                plt.xlabel("Training iteration")
                plt.ylabel("Value")
                plt.title(f"Training stats (per-iteration) - {eval_env_id or d4rl_env}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(stats_path)
                plt.close()
                print(f"Saved training stats plot to {stats_path}")
        else:
            print("matplotlib not available; skipping evaluation/training plots.")
    else:
        print("No online evaluation was performed (evaluation env not available).")
    print("=" * 60)
    print("finished training!")
    # Save final model checkpoint: use explicit path if provided,
    # otherwise default to RF/models/<env>/<env>-<dataset>_seed<seed>.pt
    save_path = args.save_model_path
    if save_path is None:
        default_dir = os.path.join("models", args.env)
        os.makedirs(default_dir, exist_ok=True)
        save_path = os.path.join(
            default_dir, f"{args.env}-{args.dataset}_seed{args.seed}.pt"
        )
    try:
        torch.save(Trainer.model.state_dict(), save_path)
        print(f"Saved model checkpoint to {save_path}")
    except Exception as e:
        print(f"Warning: failed to save model checkpoint to {save_path}: {e}")
    end_time = datetime.now().replace(microsecond=0)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("finished training at: " + end_time_str)
    print("=" * 60)

    # Final checkpoint summary.
    print(f"Model checkpoint path: {save_path}")

    # Final summary of plot directory for Slurm logs.
    print(f"All plots for this run saved under: {plot_run_dir}")

    # Plot training diagnostics: coarse per-iteration and fine-grained per-update.
    if plt is not None:
        os.makedirs(plot_run_dir, exist_ok=True)

        # Per-iteration summary (as before).
        if training_loss_list:
            iter_steps = list(range(1, len(training_loss_list) + 1))

            loss_plot_path = os.path.join(
                plot_run_dir, f"{d4rl_env}_training_losses_{start_time_str}.png"
            )
            plt.figure()
            plt.plot(iter_steps, training_loss_list, label="total_loss")
            if rtg_loss_list:
                plt.plot(iter_steps, rtg_loss_list, label="returns_to_go_loss")
            if action_loss_list:
                plt.plot(iter_steps, action_loss_list, label="action_loss")
            if temperature_loss_list:
                plt.plot(
                    iter_steps, temperature_loss_list, label="temperature_loss"
                )
            if rl_loss_list:
                plt.plot(iter_steps, rl_loss_list, label="rl_loss")
            if policy_loss_list:
                plt.plot(iter_steps, policy_loss_list, label="policy_loss")
            if critic_loss_list:
                plt.plot(iter_steps, critic_loss_list, label="critic_loss")
            plt.xlabel("Training iteration")
            plt.ylabel("Loss")
            plt.title(f"Training losses (per-iteration) - {eval_env_id or d4rl_env}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Saved per-iteration training loss plot to {loss_plot_path}")

            stats_plot_path = os.path.join(
                plot_run_dir, f"{d4rl_env}_training_stats_{start_time_str}.png"
            )
            plt.figure()
            if entropy_list:
                plt.plot(iter_steps, entropy_list, label="policy_entropy")
            if temperature_list:
                plt.plot(iter_steps, temperature_list, label="temperature")
            if grad_norm_list:
                plt.plot(iter_steps, grad_norm_list, label="grad_norm")
            plt.xlabel("Training iteration")
            plt.ylabel("Value")
            plt.title(
                f"Training stats (per-iteration) - {eval_env_id or d4rl_env}"
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(stats_plot_path)
            plt.close()
            print(f"Saved per-iteration training stats plot to {stats_plot_path}")

        # Fine-grained per-update moving-average view.
        if step_loss_list:
            # Use a window such that we get ~200 MA points.
            n_steps = len(step_loss_list)
            window = max(1, n_steps // 200)

            def moving_average(xs):
                xs_arr = np.asarray(xs, dtype=float)
                if xs_arr.shape[0] < window:
                    return xs_arr, np.arange(1, xs_arr.shape[0] + 1)
                ma = np.convolve(xs_arr, np.ones(window) / window, mode="valid")
                x = np.arange(window, window + ma.shape[0])
                return ma, x

            loss_ma, x_loss = moving_average(step_loss_list)
            rtg_ma, x_rtg = moving_average(step_rtg_loss_list) if step_rtg_loss_list else (None, None)
            act_ma, x_act = moving_average(step_action_loss_list) if step_action_loss_list else (None, None)
            temp_ma, x_temp = moving_average(step_temperature_loss_list) if step_temperature_loss_list else (None, None)
            rl_ma, x_rl = moving_average(step_rl_loss_list) if step_rl_loss_list else (None, None)
            pol_ma, x_pol = moving_average(step_policy_loss_list) if step_policy_loss_list else (None, None)
            critic_ma, x_critic = moving_average(step_critic_loss_list) if step_critic_loss_list else (None, None)

            fine_loss_plot_path = os.path.join(
                plot_run_dir, f"{d4rl_env}_training_losses_steps_{start_time_str}.png"
            )
            plt.figure()
            plt.plot(x_loss, loss_ma, label=f"total_loss_MA(w={window})")
            if rtg_ma is not None:
                plt.plot(x_rtg, rtg_ma, label="returns_to_go_loss_MA")
            if act_ma is not None:
                plt.plot(x_act, act_ma, label="action_loss_MA")
            if temp_ma is not None:
                plt.plot(x_temp, temp_ma, label="temperature_loss_MA")
            if rl_ma is not None:
                plt.plot(x_rl, rl_ma, label="rl_loss_MA")
            if pol_ma is not None:
                plt.plot(x_pol, pol_ma, label="policy_loss_MA")
            if critic_ma is not None:
                plt.plot(x_critic, critic_ma, label="critic_loss_MA")
            plt.xlabel("Update step")
            plt.ylabel("Loss (moving average)")
            plt.title(
                f"Training losses (per-update MA) - {eval_env_id or d4rl_env}"
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fine_loss_plot_path)
            plt.close()
            print(f"Saved per-update training loss plot to {fine_loss_plot_path}")

            # Policy stats: entropy, temperature, grad_norm moving averages.
            ent_ma, x_ent = moving_average(step_entropy_list) if step_entropy_list else (None, None)
            temp_val_ma, x_temp_val = moving_average(step_temperature_value_list) if step_temperature_value_list else (None, None)
            grad_ma, x_grad = moving_average(step_grad_norm_list) if step_grad_norm_list else (None, None)

            fine_stats_plot_path = os.path.join(
                plot_run_dir, f"{d4rl_env}_training_stats_steps_{start_time_str}.png"
            )
            plt.figure()
            if ent_ma is not None:
                plt.plot(x_ent, ent_ma, label="policy_entropy_MA")
            if temp_val_ma is not None:
                plt.plot(x_temp_val, temp_val_ma, label="temperature_MA")
            if grad_ma is not None:
                plt.plot(x_grad, grad_ma, label="grad_norm_MA")
            plt.xlabel("Update step")
            plt.ylabel("Value (moving average)")
            plt.title(
                f"Training stats (per-update MA) - {eval_env_id or d4rl_env}"
            )
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fine_stats_plot_path)
            plt.close()
            print(f"Saved per-update training stats plot to {fine_stats_plot_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=[ "reinformer"], default="reinformer")
    parser.add_argument("--env", type=str, default="antmaze")
    parser.add_argument("--dataset", type=str, default="medium-diverse")
    # Optional target entropy for the policy temperature update.
    # If not provided, the trainer will default to -act_dim.
    parser.add_argument("--target_entropy", type=float, default=None)
    parser.add_argument("--num_eval_ep", type=int, default=30)
    parser.add_argument("--max_eval_ep_len", type=int, default=1000)
    parser.add_argument("--dataset_dir", type=str, default="data/d4rl_dataset/")
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=256)  
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--grad_norm", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=128)  
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_train_iters", type=int, default=10)
    parser.add_argument("--num_updates_per_iter", type=int, default=5000)
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help=(
            "Optional explicit path to save the final model checkpoint. "
            "If not provided, a default under RF/models/<env>/ will be used."
        ),
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Optional path to a pretrained checkpoint to load before online finetuning.",
    )
    parser.add_argument(
        "--online_training",
        action="store_true",
        help="Enable online finetuning loop after offline training.",
    )
    parser.add_argument(
        "--beta_rl",
        type=float,
        default=0.0,
        help="Weight for the advantage-weighted RL loss during online finetuning.",
    )
    parser.add_argument(
        "--adv_scale",
        type=float,
        default=1.0,
        help="Scaling factor for advantage weighting during online finetuning.",
    )
    parser.add_argument(
        "--reinforce_online",
        action="store_true",
        help="If set, use REINFORCE-style updates during online training on new rollouts only.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "When used with --reinforce_online, use the model's RTG predictions as a baseline "
            "and fit them with an expectile (tau) critic loss."
        ),
    )
    parser.add_argument(
        "--no_buffer",
        action="store_true",
        help="When used with --reinforce_online, do not reuse a replay buffer; train only on the newly collected rollouts each iteration (batch size = num_online_rollouts_per_iter).",
    )
    parser.add_argument(
        "--online_buffer_size",
        type=int,
        default=None,
        help="Size of the replay buffer used during online finetuning; defaults to size of offline dataset.",
    )
    parser.add_argument(
        "--num_online_rollouts_per_iter",
        type=int,
        default=1,
        help="How many new online trajectories to collect each online iteration.",
    )
    parser.add_argument(
        "--rtg_gamma",
        type=float,
        default=1.0,
        help="Discount factor used when computing returns-to-go from rewards for online rollouts.",
    )
    parser.add_argument(
        "--critic_coef",
        type=float,
        default=1.0,
        help="Weight on the critic/expectile loss when using REINFORCE with a baseline.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    # use_wandb = False
    parser.add_argument("--use_wandb", action='store_true', default=False)
    args = parser.parse_args()
    
    if args.use_wandb:
        wandb.init(
            name=args.env + "-" + args.dataset,
            project="Reinformer",
            config=vars(args)
        )

    experiment(vars(args))
