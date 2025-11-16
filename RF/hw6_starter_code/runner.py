# runner.py

import argparse
import os
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
from minari.utils import get_normalized_score as minari_get_normalized_score
from td3_agent import TD3Agent
from utils import (
    evaluate_policy,
    record_eval_video,
    plot_curves,
    get_buffer_and_environments_for_task,
)


def create_agent(agent_type, env_info, offline_buffer, args, device):
    """Factory function to create the appropriate agent"""
    if agent_type.lower() == "td3":
        return TD3Agent(
            env_info=env_info,
            offline_buffer=offline_buffer,
            lr=args.lr,
            gamma=args.gamma,
            tau=args.tau,
            delay=args.delay,
            batch_size=args.batch_size,
            update_every=args.update_every,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            bc_regularization_weight=args.bc_regularization_weight,
            cql_alpha=args.cql_alpha,
            cql_n_actions=args.cql_n_actions,
            cql_temp=args.cql_temp,
            device=device,
        )
    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Choose 'ppo', 'sac', or 'td3'"
        )


def initialize_log(agent_type):
    """Initialize logging dictionary based on agent type"""
    base_log = {
        "steps": [],
        "episodic_return": [],
        "eval_normalized": [],
        "eval_mean": [],
        "eval_std": [],
        "eval_steps": [],
    }

    if agent_type.lower() == "ppo":
        base_log.update(
            {
                "loss": [],
                "policy_loss": [],
                "value_loss": [],
                "entropy": [],
                "kl": [],
                "clipfrac": [],
            }
        )
    elif agent_type.lower() in ["sac", "td3"]:
        base_log.update(
            {
                "actor_loss": [],
                "critic1_loss": [],
                "critic2_loss": [],
                "alpha": [],
                "q1": [],
                "q2": [],
                "ddpg_policy_loss": [],
                "bc_regularization_loss": [],
                "bc_regularization_weight": [],
                "cql_loss": [],
                "cql_current_actions": [],
                "in_distribution_q_pred": [],
                "cql_random_actions": [],
            }
        )

    return base_log


def _compute_normalized_score(minari_dataset, raw_return: float) -> float | None:
    """Compute D4RL-style normalized score (0-100) via Minari if available."""
    if minari_dataset is None:
        return None
    try:
        scores = minari_get_normalized_score(
            minari_dataset, np.asarray([raw_return], dtype=np.float32)
        )
        # Minari's utility returns values in [0, 1]; scale to [0, 100].
        return float(scores[0] * 100.0)
    except Exception as e:
        print(f"Warning: could not compute normalized score via Minari: {e}")
        return None


def log_training_stats(agent_type, stats, log, total_steps, updates_performed):
    """Log training statistics in a unified way"""

    if not stats:  # No update occurred
        return

    recent_returns = log["episodic_return"][-10:] if log["episodic_return"] else [0]
    avg_return = np.mean(recent_returns)

    if agent_type.lower() == "ppo":
        print(
            f"[Step {total_steps:>8d}] "
            f"Updates: {updates_performed:>4d} | "
            f"Loss: {log['loss'][-1]:>7.4f} | "
            f"π: {log['policy_loss'][-1]:>7.4f} | "
            f"V: {log['value_loss'][-1]:>7.4f} | "
            f"H: {log['entropy'][-1]:>6.3f} | "
            f"KL: {log['kl'][-1]:>8.5f} | "
            f"Clip%: {log['clipfrac'][-1]:>5.1%} | "
            f"Ret: {avg_return:>7.1f}"
        )
    elif agent_type.lower() in ["sac", "td3"]:
        print(
            f"[Step {total_steps:>8d}] "
            f"Updates: {updates_performed:>4d} | "
            f"Actor: {log['actor_loss'][-1]:>7.4f} | "
            f"C1: {log['critic1_loss'][-1]:>7.4f} | "
            f"C2: {log['critic2_loss'][-1]:>7.4f} | "
            f"α: {log['alpha'][-1]:>6.3f} | "
            f"Q1: {log['q1'][-1]:>7.2f} | "
            f"Q2: {log['q2'][-1]:>7.2f} | "
            f"DDPG: {log['ddpg_policy_loss'][-1]:>7.4f} | "
            f"BC: {log['bc_regularization_loss'][-1]:>7.4f} | "
            f"Ret: {avg_return:>7.1f}"
        )
        if "cql_loss" in stats:
            print(
                f"CQL Loss: {stats['cql_loss']:.4f} | "
                f"cql_current_actions: {stats['cql_current_actions']:.4f} | "
                f"in_distribution_q_pred: {stats['in_distribution_q_pred']:.4f} | "
                f"cql_random_actions: {stats['cql_random_actions']:.4f}"
            )


def create_transition(
    agent_type, obs, action, reward, next_obs, terminated, truncated, action_info
):
    """Create transition dict appropriate for the agent type"""
    # For off-policy methods (SAC/TD3), treat truncations as terminals to avoid
    # bootstrapping across env resets (TimeLimit). PPO handles truncation separately.
    if agent_type.lower() in ["sac", "td3"]:
        done_flag = bool(terminated or truncated)
    else:
        done_flag = bool(terminated)

    base_transition = {
        "obs": obs.copy(),
        "action": action.copy(),
        "reward": float(reward),
        "next_obs": next_obs.copy(),
        "done": done_flag,
        "truncated": bool(truncated),
    }

    if agent_type.lower() == "ppo":
        # PPO needs log_prob and value from action_info
        base_transition.update(
            {
                "log_prob": action_info.get("log_prob", 0.0),
                "value": action_info.get("value", 0.0),
            }
        )
    # SAC doesn't need extra info beyond base transition

    return base_transition


def save_final_model(agent, agent_type, out_dir):
    """Save the final model with appropriate attribute access"""
    final_model_path = os.path.join(out_dir, "final.pt")

    if agent_type.lower() == "ppo":
        if hasattr(agent, "actor"):
            torch.save(agent.actor.state_dict(), final_model_path)
        elif hasattr(agent, "policy"):
            torch.save(agent.policy.state_dict(), final_model_path)
        else:
            print("Warning: Could not find policy to save for PPO agent")
            return
    elif agent_type.lower() in ["sac", "td3"]:
        torch.save(agent.actor.state_dict(), final_model_path)

    print(f"Final model saved: {final_model_path}")


def run(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    # Environment setup
    offline_buffer, eval_env, minari_dataset = get_buffer_and_environments_for_task(
        minari_dataset_name=args.env_id, device=device
    )
    print(f"Environment: {args.env_id}")

    # Environment spaces
    assert isinstance(eval_env.observation_space, gym.spaces.Box)
    assert isinstance(eval_env.action_space, gym.spaces.Box)
    obs_dim = int(np.prod(eval_env.observation_space.shape))
    act_dim = int(np.prod(eval_env.action_space.shape))
    env_info = {
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "act_low": eval_env.action_space.low,
        "act_high": eval_env.action_space.high,
    }
    print(f"Obs dim: {obs_dim}, Action dim: {act_dim}")

    # Create agent
    agent = create_agent(args.agent, env_info, offline_buffer, args, device)
    print(f"Created {args.agent.upper()} agent")

    # Setup output directories
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    # Initialize logging
    log = initialize_log(args.agent)

    # Training state
    obs, _ = eval_env.reset(seed=args.seed)
    total_steps = 0
    best_eval = -1e9
    updates_performed = 0

    print(f"Starting {args.agent.upper()} training...")
    print(f"Target steps: {args.total_steps}")
    if args.agent.lower() == "ppo":
        print(f"Rollout steps: {args.rollout_steps}")
        print(f"Update epochs: {args.update_epochs}")
        print(f"Minibatch size: {args.minibatch_size}")
    else:  # SAC
        print(f"Batch size: {args.batch_size}")
        print(f"Update every: {args.update_every}")
    print("-" * 50)

    last_stat_update = 0
    while total_steps < args.total_steps:

        # Step the agent
        stats = agent.step()
        total_steps += 1

        # Collect optimization stats when update occurred
        if stats:  # stats is non-empty dict when update happened
            updates_performed += 1
            # Add all stats to log (different agents have different metrics)
            for k, v in stats.items():
                if k in log:
                    log[k].append(v)

        if stats:
            if total_steps - last_stat_update >= args.log_every:
                log_training_stats(
                    args.agent, stats, log, total_steps, updates_performed
                )
                last_stat_update = total_steps

        # Periodic evaluation
        if args.eval_every > 0 and total_steps % args.eval_every == 0:
            print(f"\n--- Evaluation at step {total_steps} ---")
            with torch.no_grad():
                mean_r, std_r = evaluate_policy(
                    agent,
                    env=eval_env,
                    episodes=args.eval_episodes,
                    seed=1000,
                    device=device,
                )
            norm_score = _compute_normalized_score(minari_dataset, mean_r)
            log["eval_mean"].append(mean_r)
            log["eval_std"].append(std_r)
            log["eval_steps"].append(total_steps)
            if norm_score is not None:
                log["eval_normalized"].append(norm_score)
                print(
                    f"Eval: {mean_r:.1f} ± {std_r:.1f} "
                    f"(normalized: {norm_score:.1f})"
                )
            else:
                print(f"Eval: {mean_r:.1f} ± {std_r:.1f}")

            # Save best model
            if mean_r > best_eval:
                best_eval = mean_r
                model_path = os.path.join(args.out_dir, "best.pt")
                # Save appropriate model part
                if hasattr(agent, "actor"):
                    torch.save(agent.actor.state_dict(), model_path)
                elif hasattr(agent, "policy"):
                    torch.save(agent.policy.state_dict(), model_path)
                print(f"New best model saved! Return: {mean_r:.1f}")
            print("--- End Evaluation ---\n")

        # Periodic video recording
        if args.video_every > 0 and total_steps % args.video_every == 0:
            print(f"Recording video at step {total_steps}...")
            name = f"{args.video_prefix}_t{total_steps}"
            try:
                with torch.no_grad():
                    path = record_eval_video(
                        agent,
                        eval_env,
                        video_dir=args.video_dir,
                        video_name=name,
                        episodes=1,
                        device=device,
                    )
                print(f"Video saved: {path}")
            except Exception as e:
                print(f"Video recording failed: {e}")
                import traceback

                traceback.print_exc()

        # Periodic plot refresh
        if args.plot_every > 0 and total_steps % args.plot_every == 0:
            if any(
                len(log.get(k, [])) > 0
                for k in log.keys()
                if k not in ["eval_mean", "eval_std", "eval_steps"]
            ):
                try:
                    plot_path = os.path.join(args.out_dir, "training_curves.png")
                    plot_curves(log, out_path=plot_path)
                    # print(f"Training plots updated: {plot_path}")
                except Exception as e:
                    print(f"Plotting failed: {e}")
                    import traceback

                    traceback.print_exc()

    # Final cleanup and saving
    print(f"\nTraining completed! Total steps: {total_steps}")
    print(f"Total {args.agent.upper()} updates: {updates_performed}")

    # Final evaluation
    if args.eval_episodes > 0:
        # load best model for final evaluation
        if best_eval > -1e9:
            model_path = os.path.join(args.out_dir, "best.pt")
            if os.path.exists(model_path):
                if hasattr(agent, "actor"):
                    agent.actor.load_state_dict(
                        torch.load(model_path, map_location=device)
                    )
                elif hasattr(agent, "policy"):
                    agent.policy.load_state_dict(
                        torch.load(model_path, map_location=device)
                    )
                print(f"Loaded best model from {model_path} for final evaluation")
            else:
                print(
                    f"Best model file not found at {model_path}, using current model for final eval"
                )

        print("Final evaluation...")
        with torch.no_grad():
            mean_r, std_r = evaluate_policy(
                agent,
                env=eval_env,
                episodes=args.eval_episodes * 2,  # More episodes for final eval
                seed=4200,
                device=device,
            )
        final_norm = _compute_normalized_score(minari_dataset, mean_r)
        if final_norm is not None:
            print(
                f"Final performance: {mean_r:.1f} ± {std_r:.1f} "
                f"(normalized: {final_norm:.1f})"
            )
            log["eval_normalized"].append(final_norm)
        else:
            print(f"Final performance: {mean_r:.1f} ± {std_r:.1f}")
        log["eval_mean"].append(mean_r)
        log["eval_std"].append(std_r)
        log["eval_steps"].append(total_steps)

    # Save final artifacts
    try:
        plot_path = os.path.join(args.out_dir, "training_curves.png")
        plot_curves(log, out_path=plot_path)
        print(f"Final training curves saved: {plot_path}")
    except Exception as e:
        print(f"Final plotting failed: {e}")
        import traceback

        traceback.print_exc()

    # Save final model
    save_final_model(agent, args.agent, args.out_dir)

    # Save training log as JSON for analysis
    import json

    log_path = os.path.join(args.out_dir, "training_log.json")
    # Convert numpy types to Python types for JSON serialization
    json_log = {}
    for k, v in log.items():
        if isinstance(v, list):
            json_log[k] = [
                float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v
            ]
        else:
            json_log[k] = v

    with open(log_path, "w") as f:
        json.dump(json_log, f, indent=2)
    print(f"Training log saved: {log_path}")

    print(f"\nAll artifacts saved to: {args.out_dir}")
    print("Training complete!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unified PPO/SAC Training")

    # Algorithm selection
    p.add_argument(
        "--agent",
        type=str,
        choices=["td3"],
        default="td3",
        help="RL algorithm to use",
    )

    # Environment
    p.add_argument(
        "--env_id",
        type=str,
        default="mujoco/hopper/medium-v0",
        help="Gym environment ID",
    )

    # Training
    p.add_argument(
        "--total_steps", type=int, default=100_000, help="Total environment steps"
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU usage")
    p.add_argument("--seed", type=int, default=42000, help="Random seed")

    # Common hyperparameters
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # PPO-specific hyperparameters
    p.add_argument(
        "--rollout_steps", type=int, default=4096, help="Steps per PPO rollout"
    )
    p.add_argument(
        "--update_epochs", type=int, default=10, help="PPO update epochs per rollout"
    )
    p.add_argument(
        "--minibatch_size", type=int, default=128, help="Minibatch size for PPO updates"
    )
    p.add_argument("--gae_lambda", type=float, default=0.98, help="GAE lambda")
    p.add_argument("--clip_coef", type=float, default=0.2, help="PPO clip coefficient")
    p.add_argument(
        "--vf_coef", type=float, default=0.5, help="Value function coefficient"
    )
    p.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="Max gradient norm for clipping",
    )

    # SAC-specific hyperparameters
    p.add_argument(
        "--tau", type=float, default=0.005, help="Target network soft update rate"
    )
    p.add_argument("--alpha", type=float, default=0.2, help="SAC entropy coefficient")
    p.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for SAC updates"
    )
    p.add_argument(
        "--update_every", type=int, default=1, help="Update frequency (every N steps)"
    )
    p.add_argument(
        "--utd_ratio",
        type=int,
        default=1,
        help="SAC updates per env step when updates occur (UTD ratio)",
    )
    # TD3-specific noise hyperparameters
    p.add_argument(
        "--policy_noise",
        type=float,
        default=0.2,
        help="TD3 target policy smoothing noise std",
    )
    p.add_argument("--delay", type=float, default=2, help="TD3 target policy delay")
    p.add_argument(
        "--noise_clip", type=float, default=0.5, help="TD3 target policy noise clip"
    )
    p.add_argument(
        "--exploration_noise",
        type=float,
        default=0.1,
        help="TD3 exploration noise std during env interaction",
    )
    p.add_argument(
        "--bc_regularization_weight",
        type=float,
        default=0.0,
        help="BC regularization weight",
    )
    p.add_argument(
        "--cql_alpha",
        type=float,
        default=0.0,
        help="CQL alpha value",
    )
    p.add_argument(
        "--cql_n_actions",
        type=int,
        default=4,
        help="Number of actions to sample for CQL",
    )
    p.add_argument(
        "--cql_temp",
        type=float,
        default=1.0,
        help="CQL temperature",
    )

    # Logging and evaluation
    p.add_argument(
        "--log_every",
        type=int,
        default=10_000,
        help="Print training stats every N steps",
    )
    p.add_argument(
        "--plot_every", type=int, default=1_000, help="Update plots every N steps"
    )
    p.add_argument(
        "--eval_every",
        type=int,
        default=50_000,
        help="Run evaluation every N steps (0 to disable)",
    )
    p.add_argument(
        "--eval_episodes", type=int, default=100, help="Episodes per evaluation"
    )

    # Video recording
    p.add_argument(
        "--video_every",
        type=int,
        default=100_000,
        help="Record video every N steps (0 to disable)",
    )
    p.add_argument(
        "--video_dir", type=str, default="videos", help="Video output directory"
    )
    p.add_argument(
        "--video_prefix", type=str, default="eval", help="Video filename prefix"
    )

    # Output
    p.add_argument(
        "--out_dir",
        type=str,
        default="runs/rl_training",
        help="Output directory for models and logs",
    )

    args = p.parse_args()

    # Build base output directory
    if args.out_dir == "runs/rl_training":
        base_out_dir = f"runs/{args.agent}_{args.env_id.lower().replace('-', '_')}"
    else:
        base_out_dir = args.out_dir

    # Append timestamp to avoid overwriting previous runs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out_dir = os.path.join(base_out_dir, timestamp)

    # Place videos inside this run directory
    args.video_dir = os.path.join(args.out_dir, "videos")

    # Print configuration
    print("=" * 60)
    print(f"{args.agent.upper()} Training Configuration")
    print("=" * 60)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:20s}: {value}")
    print("=" * 60)

    run(args)
