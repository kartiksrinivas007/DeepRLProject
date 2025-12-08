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
        return discount_cumsum_np(rewards_np, gamma=1.0) / reward_scale

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

    if not variant.get("online_training", False):
        # Optional initial evaluation before any training updates.
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

        # Progress bar over training iterations to indicate offline training progress.
        for itr in tqdm(range(1, max_train_iters + 1), desc="Training iterations"):
            t1 = time.time()
            for epoch in range(num_updates_per_iter):
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

                loss = Trainer.train_step(
                    timesteps=timesteps,
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go,
                    rewards=rewards,
                    traj_mask=traj_mask
                )
                if args.use_wandb:
                    wandb.log(
                        data={
                            "training/loss" : loss,
                        }
                    )
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
        # Online finetuning loop with advantage-weighted RL term.
        online_buffer_size = variant.get("online_buffer_size", len(traj_dataset))
        beta_rl = variant.get("beta_rl", 0.0)
        adv_scale = variant.get("adv_scale", 1.0)

        # Initialize buffer with top offline trajectories by return.
        offline_returns = [
            float(np.sum(traj["rewards"])) for traj in traj_dataset.trajectories
        ]
        sorted_indices = np.argsort(offline_returns)[::-1]
        online_replay_buffer = [
            traj_dataset.trajectories[i] for i in sorted_indices[:online_buffer_size]
        ]

        # Optionally load a pretrained model checkpoint.
        if variant.get("pretrained_model"):
            try:
                state_dict = torch.load(
                    variant["pretrained_model"], map_location=device
                )
                Trainer.model.load_state_dict(state_dict)
                print(f"Loaded pretrained model from {variant['pretrained_model']}")
            except Exception as e:
                print(f"Warning: could not load pretrained model: {e}")

        try:
            rollout_env = gym.make(d4rl_env)
            rollout_env.reset(seed=seed + 1 if seed is not None else None)
        except Exception:
            rollout_env = eval_env

        for itr in tqdm(range(1, max_train_iters + 1), desc="Online iterations"):
            # 1) Collect one rollout with stochastic policy (rsample).
            new_traj = collect_online_trajectory(
                model=Trainer.model,
                env_for_rollout=rollout_env,
                use_mean_action=False,
            )
            online_replay_buffer.append(new_traj)
            if len(online_replay_buffer) > online_buffer_size:
                online_replay_buffer.pop(0)

            # 2) Train for num_updates_per_iter minibatches from buffer.
            t1 = time.time()
            for _ in range(num_updates_per_iter):
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

                loss = Trainer.train_step(
                    timesteps=timesteps,
                    states=states,
                    actions=actions,
                    returns_to_go=returns_to_go,
                    rewards=rewards,
                    traj_mask=traj_mask,
                    beta_rl=beta_rl,
                    adv_scale=adv_scale,
                )
                if args.use_wandb:
                    wandb.log(
                        data={
                            "training/loss": loss,
                        }
                    )
            t2 = time.time()

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

    if normalized_d4rl_score_list and args.use_wandb:
        wandb.log(
            data={
                "evaluation/max_score": max(normalized_d4rl_score_list),
                "evaluation/last_score": normalized_d4rl_score_list[-1],
            }
        )
    if normalized_d4rl_score_list:
        print("Evaluation scores over iterations:", normalized_d4rl_score_list)

        if raw_return_list:
            print("Raw returns over iterations:", raw_return_list)

        # Plot evaluation curves if matplotlib is available.
        if plt is not None:
            os.makedirs("plots", exist_ok=True)
            # Use start_time_str to avoid overwriting previous runs.
            plot_path_norm = os.path.join(
                "plots", f"{d4rl_env}_eval_{start_time_str}.png"
            )
            plt.figure()
            plt.plot(normalized_d4rl_score_list, marker="o")
            plt.xlabel("Training iteration")
            plt.ylabel("Normalized score")
            plt.title(f"Online evaluation - {eval_env_id or d4rl_env}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path_norm)
            plt.close()

            # Raw return plot.
            if raw_return_list:
                plot_path_raw = os.path.join(
                    "plots", f"{d4rl_env}_eval_raw_return_{start_time_str}.png"
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
        else:
            print("matplotlib not available; skipping evaluation plot.")
    else:
        print("No online evaluation was performed (evaluation env not available).")
    print("=" * 60)
    print("finished training!")
    # Optional checkpoint save
    if args.save_model_path is not None:
        try:
            torch.save(Trainer.model.state_dict(), args.save_model_path)
            print(f"Saved model checkpoint to {args.save_model_path}")
        except Exception as e:
            print(f"Warning: failed to save model checkpoint to {args.save_model_path}: {e}")
    end_time = datetime.now().replace(microsecond=0)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("finished training at: " + end_time_str)
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=[ "reinformer"], default="reinformer")
    parser.add_argument("--env", type=str, default="antmaze")
    parser.add_argument("--dataset", type=str, default="medium-diverse")
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--online_training", action="store_true", default=False)
    parser.add_argument("--online_buffer_size", type=int, default=1000)
    parser.add_argument("--beta_rl", type=float, default=0.0, help="Weight for AWAC-style RL fine-tuning loss during online training.")
    parser.add_argument("--adv_scale", type=float, default=1.0, help="Lambda scale for advantage weighting exp(A/lambda).")
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--save_model_path", type=str, default=None, help="Optional path to save model state_dict after training.")
    parser.add_argument("--target_entropy", type=float, default=None, help="Optional target entropy for the stochastic policy; defaults to -action_dim when unset.")
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
