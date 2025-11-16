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
    if env == "kitchen":
        d4rl_env = f"{env}-{dataset}-v0"
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

    # Try to create an evaluation environment if available.
    eval_env = None
    eval_env_id = None
    eval_dataset_for_norm = None
    use_minari_norm = False

    # For AntMaze and Hopper, prefer the Minari-provided evaluation environment.
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
    elif env == "hopper" and dataset == "medium":
        try:
            minari_id = "mujoco/hopper/medium-v0"
            ds = minari.load_dataset(minari_id)
            eval_dataset_for_norm = ds
            eval_env = ds.recover_environment(
                eval_env=True, max_episode_steps=variant["max_eval_ep_len"]
            )
            eval_env.reset(seed=seed)
            env_spec = ds._eval_env_spec or ds.spec.env_spec
            eval_env_id = env_spec.id if env_spec is not None else d4rl_env
            # Hopper Minari datasets do not provide ref_min_score/ref_max_score,
            # so we do not use Minari normalization here.
            use_minari_norm = False
            print(f"Using Minari eval environment: {eval_env_id}")
        except Exception as e:
            print(
                f"Warning: could not recover Hopper evaluation env from Minari. "
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
                # Prefer Minari normalization if reference scores are available.
                if use_minari_norm and eval_dataset_for_norm is not None:
                    try:
                        norm_score = minari.get_normalized_score(
                            eval_dataset_for_norm, np.array([return_mean])
                        )[0]
                        return norm_score * 100.0
                    except ValueError:
                        # Dataset does not have reference scores; fall through.
                        pass
                # For dense Mujoco tasks like hopper, compute D4RL-style
                # normalized score manually using REF_MIN/REF_MAX from d4rl.
                if env == "hopper" and dataset == "medium":
                    try:
                        from d4rl import infos as d4rl_infos

                        ref_min = d4rl_infos.REF_MIN_SCORE.get(
                            "hopper-medium-v0",
                            d4rl_infos.REF_MIN_SCORE["hopper-random-v0"],
                        )
                        ref_max = d4rl_infos.REF_MAX_SCORE.get(
                            "hopper-medium-v0",
                            d4rl_infos.REF_MAX_SCORE["hopper-random-v0"],
                        )
                        norm = (return_mean - ref_min) / (ref_max - ref_min)
                        return norm * 100.0
                    except Exception:
                        pass
                # Next, try D4RL-style env normalization if available.
                if hasattr(eval_env, "get_normalized_score"):
                    return eval_env.get_normalized_score(return_mean) * 100.0
                # Fallback: return the raw average return.
                return return_mean

    max_train_iters = variant["max_train_iters"]
    num_updates_per_iter = variant["num_updates_per_iter"]
    normalized_d4rl_score_list = []

    # Optional initial evaluation before any training updates.
    if evaluator is not None:
        print("Running initial evaluation before training...")
        init_score = evaluator(model=Trainer.model)
        normalized_d4rl_score_list.append(init_score)
        if args.use_wandb:
            wandb.log(
                data={
                    "evaluation/score": init_score,
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
            normalized_d4rl_score = evaluator(model=Trainer.model)
            t3 = time.time()
            normalized_d4rl_score_list.append(normalized_d4rl_score)
            if args.use_wandb:
                wandb.log(
                    data={
                        "training/time": t2 - t1,
                        "evaluation/score": normalized_d4rl_score,
                        "evaluation/time": t3 - t2,
                    }
                )

        # Simple textual indicator that offline training is progressing.
        print(
            f"Iteration {itr}/{max_train_iters} - "
            f"last training loss: {loss:.4f}"
            + (
                f", last eval score: {normalized_d4rl_score:.2f}"
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

        # Plot evaluation curve if matplotlib is available.
        if plt is not None:
            os.makedirs("plots", exist_ok=True)
            plot_path = os.path.join("plots", f"{d4rl_env}_eval.png")
            plt.figure()
            plt.plot(normalized_d4rl_score_list, marker="o")
            plt.xlabel("Training iteration")
            plt.ylabel("Normalized score")
            plt.title(f"Online evaluation - {eval_env_id or d4rl_env}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved evaluation plot to {plot_path}")
        else:
            print("matplotlib not available; skipping evaluation plot.")
    else:
        print("No online evaluation was performed (evaluation env not available).")
    print("=" * 60)
    print("finished training!")
    end_time = datetime.now().replace(microsecond=0)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("finished training at: " + end_time_str)
    print("=" * 60)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=[ "reinformer"], default="reinformer")
    parser.add_argument("--env", type=str, default="antmaze")
    parser.add_argument("--dataset", type=str, default="medium-diverse")
    parser.add_argument("--num_eval_ep", type=int, default=10)
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
