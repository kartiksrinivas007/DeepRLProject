import gymnasium as gym
import numpy as np
import torch
import wandb
import minari

import argparse
import pickle
import random
import sys
import os
import pathlib
from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def compute_normalized_score(
    raw_return,
    env,
    env_name,
    dataset,
    eval_dataset_for_norm=None,
):
    normalized_score = None

    # Prefer Minari normalization if reference scores are available.
    if eval_dataset_for_norm is not None:
        try:
            norm_score = minari.get_normalized_score(
                eval_dataset_for_norm, np.array([raw_return])
            )[0]
            normalized_score = norm_score * 100.0
        except Exception:
            pass

    # For classic D4RL Mujoco tasks, optionally fall back to REF_MIN/REF_MAX
    # if d4rl is installed (mirrors Reinformer logic).
    if (
        normalized_score is None
        and env_name in ["hopper", "walker2d", "halfcheetah"]
        and dataset == "medium"
    ):
        try:
            from d4rl import infos as d4rl_infos  # type: ignore

            key_medium = f"{env_name}-medium-v0"
            key_random = f"{env_name}-random-v0"
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

    # Next, try env-provided normalization if available.
    if normalized_score is None and hasattr(env, "get_normalized_score"):
        try:
            normalized_score = env.get_normalized_score(raw_return) * 100.0
        except Exception:
            pass

    # Fallback: if normalization still not available, just use raw return.
    if normalized_score is None:
        normalized_score = raw_return

    return normalized_score


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    model_dir = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        f'./models/{env_name}/',
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    eval_dataset_for_norm = None
    # Match Reinformer / gym-DT style: prefer Minari eval env, fall back to
    # Gymnasium, and use the same reward scaling convention.
    if env_name == 'hopper':
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets (raw returns)
        scale = 1000.  # reward / RTG normalization as in Reinformer
        env = None
        try:
            minari_id = f"mujoco/hopper/{dataset}-v0"
            ds = minari.load_dataset(minari_id)
            eval_dataset_for_norm = ds
            env = ds.recover_environment(
                eval_env=True,
                max_episode_steps=max_ep_len,
            )
            env.reset()
            env_id = getattr(env, "spec", None).id if getattr(env, "spec", None) else "unknown"
            print(f"Using Minari eval environment from '{minari_id}' (env id: {env_id})")
        except Exception as e:
            print(
                f"Warning: could not recover Hopper eval env from Minari. "
                f"Falling back to Gymnasium make. Error: {e}"
            )
            for env_id in ['Hopper-v5', 'Hopper-v4', 'Hopper-v3']:
                try:
                    env = gym.make(env_id)
                    print(f"Using Gymnasium eval environment: {env_id}")
                    break
                except Exception:
                    continue
        if env is None:
            raise RuntimeError("Could not create Hopper evaluation environment.")
    elif env_name == 'halfcheetah':
        max_ep_len = 1000
        env_targets = [12000, 6000]
        # Use same reward scaling as Reinformer: larger scale for HalfCheetah.
        scale = 5000.
        env = None
        try:
            minari_id = f"mujoco/halfcheetah/{dataset}-v0"
            ds = minari.load_dataset(minari_id)
            eval_dataset_for_norm = ds
            env = ds.recover_environment(
                eval_env=True,
                max_episode_steps=max_ep_len,
            )
            env.reset()
            env_id = getattr(env, "spec", None).id if getattr(env, "spec", None) else "unknown"
            print(f"Using Minari eval environment from '{minari_id}' (env id: {env_id})")
        except Exception as e:
            print(
                f"Warning: could not recover HalfCheetah eval env from Minari. "
                f"Falling back to Gymnasium make. Error: {e}"
            )
            for env_id in ['HalfCheetah-v5', 'HalfCheetah-v4', 'HalfCheetah-v3']:
                try:
                    env = gym.make(env_id)
                    print(f"Using Gymnasium eval environment: {env_id}")
                    break
                except Exception:
                    continue
        if env is None:
            raise RuntimeError("Could not create HalfCheetah evaluation environment.")
    elif env_name == 'walker2d':
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
        env = None
        try:
            minari_id = f"mujoco/walker2d/{dataset}-v0"
            ds = minari.load_dataset(minari_id)
            eval_dataset_for_norm = ds
            env = ds.recover_environment(
                eval_env=True,
                max_episode_steps=max_ep_len,
            )
            env.reset()
            env_id = getattr(env, "spec", None).id if getattr(env, "spec", None) else "unknown"
            print(f"Using Minari eval environment from '{minari_id}' (env id: {env_id})")
        except Exception as e:
            print(
                f"Warning: could not recover Walker2d eval env from Minari. "
                f"Falling back to Gymnasium make. Error: {e}"
            )
            for env_id in ['Walker2d-v5', 'Walker2d-v4', 'Walker2d-v3']:
                try:
                    env = gym.make(env_id)
                    print(f"Using Gymnasium eval environment: {env_id}")
                    break
                except Exception:
                    continue
        if env is None:
            raise RuntimeError("Could not create Walker2d evaluation environment.")
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError
    
    # Override env_targets / set different training target for online decision transformer, following paper
    if variant['online_training']:
        if env_name == 'hopper':
            env_targets = [3600]  # evaluation conditioning targets
            target_online = 7200
        elif env_name == 'halfcheetah':
            env_targets = [6000]
            target_online = 12000
        elif env_name == 'walker2d':
            env_targets = [5000]
            target_online = 10000
        else:
            raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    
    # Sort trajectories from worst to best and cut to buffer size
    if variant['online_training']:
        trajectories = [trajectories[index] for index in sorted_inds]
        trajectories = trajectories[:variant['online_buffer_size']]
        num_trajectories = len(trajectories)

    starting_p_sample = p_sample
    def get_batch(batch_size=256, max_len=K):
        # Dynamically recompute p_sample if online training
        if variant['online_training']:
            traj_lens = np.array([len(path['observations']) for path in trajectories])
            p_sample = traj_lens / sum(traj_lens)
        else:
            p_sample = starting_p_sample
            
        
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            if variant['online_training']:
                traj = trajectories[batch_inds[i]]
            else:
                traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * 0., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    if variant['online_training']:
        # If online training, use means during eval, but (not during exploration)
        variant['use_action_means'] = True
    
    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            use_means=variant['use_action_means'],
                            eval_context=variant['eval_context']
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            # Compute normalized scores per episode using the same scheme as
            # Reinformer (Minari / D4RL / env.get_normalized_score).
            norm_scores = [
                compute_normalized_score(
                    raw_return,
                    env,
                    env_name,
                    dataset,
                    eval_dataset_for_norm=eval_dataset_for_norm,
                )
                for raw_return in returns
            ]
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'target_{target_rew}_normalized_return_mean': np.mean(norm_scores),
                f'target_{target_rew}_normalized_return_std': np.std(norm_scores),
            }
        return fn


    if model_type == 'dt':
        if variant['pretrained_model']:
            # Torch 2.6 defaults weights_only=True; the pretrained DT checkpoints
            # are full objects, so explicitly allow loading the class.
            try:
                torch.serialization.add_safe_globals([DecisionTransformer])
            except Exception:
                # Older torch versions may not have add_safe_globals; fallback works with weights_only=False.
                pass
            model = torch.load(
                variant['pretrained_model'],
                map_location=device,
                weights_only=False,
            )
            model.stochastic_tanh = variant['stochastic_tanh']
            model.approximate_entropy_samples = variant['approximate_entropy_samples']
            model.to(device)

        else:
            model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                max_ep_len=max_ep_len*2,
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
                n_head=variant['n_head'],
                n_inner=4*variant['embed_dim'],
                activation_function=variant['activation_function'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
                stochastic = variant['stochastic'],
                remove_pos_embs=variant['remove_pos_embs'],
                approximate_entropy_samples = variant['approximate_entropy_samples'],
                stochastic_tanh=variant['stochastic_tanh']
            )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
        
    if variant['online_training']:
        assert(variant['pretrained_model'] is not None), "Must specify pretrained model to perform online finetuning"
        variant['use_entropy'] = True
        
    if variant['online_training'] and variant['target_entropy']:
        # Setup variable and optimizer for (log of) lagrangian multiplier used for entropy constraint
        # We optimize the log of the multiplier b/c lambda >= 0
        log_entropy_multiplier = torch.zeros(1, requires_grad=True, device=device)
        multiplier_optimizer = torch.optim.AdamW(
            [log_entropy_multiplier],
            lr=variant['learning_rate'],
            weight_decay=variant['weight_decay'],
        )
        # multiplier_optimizer = torch.optim.Adam(
        #     [log_entropy_multiplier],
        #     lr=1e-3
        #     #lr=variant['learning_rate'],
        # )
        multiplier_scheduler = torch.optim.lr_scheduler.LambdaLR(
            multiplier_optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
    else:
        log_entropy_multiplier = None
        multiplier_optimizer = None 
        multiplier_scheduler = None

    entropy_loss_fn = None
    if variant['stochastic']:
        if variant['use_entropy']:
            if variant['target_entropy']:
                loss_fn = lambda s_hat, a_hat, rtg_hat,r_hat, s, a, rtg, r, a_log_prob, entropies: -torch.mean(a_log_prob) - torch.exp(log_entropy_multiplier.detach()) * torch.mean(entropies)
                target_entropy = -act_dim
                entropy_loss_fn = lambda entropies: torch.exp(log_entropy_multiplier) * (torch.mean(entropies.detach()) - target_entropy)
            else:
                loss_fn = lambda s_hat, a_hat, rtg_hat,r_hat, s, a, rtg, r, a_log_prob, entropies: -torch.mean(a_log_prob) - torch.mean(entropies)
        else:
            loss_fn = lambda s_hat, a_hat, rtg_hat, r_hat, s, a, rtg,r, a_log_prob, entropies: -torch.mean(a_log_prob)
    else:
        loss_fn = lambda s_hat, a_hat, rtg_hat, r_hat, s, a, rtg, r, a_log_prob, entropies: torch.mean((a_hat - a)**2)

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=loss_fn,
            log_entropy_multiplier=log_entropy_multiplier,
            entropy_loss_fn=entropy_loss_fn,
            multiplier_optimizer=multiplier_optimizer,
            multiplier_scheduler=multiplier_scheduler,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=loss_fn,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug
    if variant['eval_only']:
        model.eval()
        eval_fns = [eval_episodes(tar) for tar in env_targets]
        
        for iter_num in range(variant['max_iters']):
            logs = {}
            for eval_fn in eval_fns:
                outputs = eval_fn(model)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v

            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
    else:
        # Track evaluation returns (raw and normalized) over training iterations for each target.
        eval_return_history = {tar: [] for tar in env_targets}
        eval_return_history_norm = {tar: [] for tar in env_targets}

        if variant['online_training']:
            for iter in range(variant['max_iters']):
                # Collect new rollout, using stochastic policy
                ret, length, traj = evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_online / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    use_means=False,
                    return_traj=True,
                )
                # Remove oldest trajectory, add new trajectory
                trajectories = trajectories[1:]
                trajectories.append(traj)

                # Perform update, eval using deterministic policy
                outputs = trainer.train_iteration(
                    num_steps=variant['num_steps_per_iter'],
                    iter_num=iter + 1,
                    print_logs=True,
                )
                # Accumulate evaluation means per target over iterations.
                for tar in env_targets:
                    key_raw = f"evaluation/target_{tar}_return_mean"
                    key_norm = f"evaluation/target_{tar}_normalized_return_mean"
                    if key_raw in outputs:
                        eval_return_history[tar].append(outputs[key_raw])
                    if key_norm in outputs:
                        eval_return_history_norm[tar].append(outputs[key_norm])
                if log_to_wandb:
                    wandb.log(outputs)
        else:
            for iter in range(variant['max_iters']):
                outputs = trainer.train_iteration(
                    num_steps=variant['num_steps_per_iter'],
                    iter_num=iter + 1,
                    print_logs=True,
                )
                for tar in env_targets:
                    key_raw = f"evaluation/target_{tar}_return_mean"
                    key_norm = f"evaluation/target_{tar}_normalized_return_mean"
                    if key_raw in outputs:
                        eval_return_history[tar].append(outputs[key_raw])
                    if key_norm in outputs:
                        eval_return_history_norm[tar].append(outputs[key_norm])
                if log_to_wandb:
                    wandb.log(outputs)

        # After training, plot evaluation raw and normalized returns vs training iterations.
        if plt is not None and any(len(v) > 0 for v in eval_return_history.values()):
            plots_dir = os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "plots",
            )
            os.makedirs(plots_dir, exist_ok=True)
            # Use the experiment prefix (which already includes a random id)
            # to make plot filenames unique per run.
            plot_id = exp_prefix
            steps = None
            for tar, vals in eval_return_history.items():
                if steps is None:
                    steps = list(range(1, len(vals) + 1))
                # If some targets have fewer points (e.g., early failures),
                # align steps per-target.
            # Raw returns
            plt.figure()
            for tar, vals in eval_return_history.items():
                if len(vals) == 0:
                    continue
                x = list(range(1, len(vals) + 1))
                plt.plot(x, vals, marker="o", label=f"target_{tar}")
            plt.xlabel("Training iteration")
            plt.ylabel("Evaluation return (raw)")
            plt.title(f"Eval returns over iterations - {env_name} {dataset} ({model_type})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plot_path_raw = os.path.join(
                plots_dir,
                f"{plot_id}_{model_type}_{env_name}-{dataset}_eval_returns_raw.png",
            )
            plt.savefig(plot_path_raw)
            plt.close()

            # Normalized returns (if available)
            if any(len(v) > 0 for v in eval_return_history_norm.values()):
                plt.figure()
                for tar, vals in eval_return_history_norm.items():
                    if len(vals) == 0:
                        continue
                    x = list(range(1, len(vals) + 1))
                    plt.plot(x, vals, marker="o", label=f"target_{tar}")
                plt.xlabel("Training iteration")
                plt.ylabel("Evaluation return (normalized score)")
                plt.title(f"Eval returns (normalized) - {env_name} {dataset} ({model_type})")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plot_path_norm = os.path.join(
                    plots_dir,
                    f"{plot_id}_{model_type}_{env_name}-{dataset}_eval_returns_normalized.png",
                )
                plt.savefig(plot_path_norm)
                plt.close()

        torch.save(model, os.path.join(model_dir, model_type + "_" + exp_prefix + ".pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--pretrained_model', default=None, type=str)
    parser.add_argument('--stochastic', default=False, action='store_true')
    parser.add_argument('--use_entropy', default=False, action='store_true')
    parser.add_argument('--use_action_means', default=False, action='store_true')
    parser.add_argument('--online_training', default=False, action='store_true')
    parser.add_argument('--online_buffer_size', default=1000, type=int) # keep top N trajectories for online training in replay buffer to start
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--remove_pos_embs', default=False, action='store_true')
    parser.add_argument('--eval_context', default=None, type=int)
    parser.add_argument('--target_entropy', default=False, action='store_true')
    parser.add_argument('--stochastic_tanh', default=False, action='store_true')
    parser.add_argument('--approximate_entropy_samples',default=1000, type=int, help="if using stochastic network w/ tanh squashing, have to approximate entropy with k samples, as no anlytical solution")
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
