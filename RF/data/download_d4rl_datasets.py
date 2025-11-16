import os
import pickle
from typing import Dict, List

import minari
import numpy as np


def _episode_to_path(episode) -> Dict[str, np.ndarray]:
    """Convert a Minari EpisodeData to our trajectory dict format.

    For AntMaze (and other goal-based tasks), Minari stores observations as a
    dict with keys like ``observation``, ``achieved_goal``, ``desired_goal``.
    We use only the low-level ``observation`` component to match the original
    D4RL state used by this codebase.
    """
    num_steps = len(episode.rewards)

    obs_raw = episode.observations
    # AntMaze: observations is a dict; we take the main 'observation' key.
    if isinstance(obs_raw, dict):
        if "observation" not in obs_raw:
            raise ValueError(
                f"Dict observations missing 'observation' key: {list(obs_raw.keys())}"
            )
        obs = np.asarray(obs_raw["observation"])
    else:
        obs = np.asarray(obs_raw)
    acts = np.asarray(episode.actions)
    rewards = np.asarray(episode.rewards, dtype=np.float32)
    terminations = np.asarray(episode.terminations, dtype=np.bool_)
    truncations = np.asarray(episode.truncations, dtype=np.bool_)

    if obs.ndim < 1:
        raise ValueError(
            f"Unsupported observation structure with shape {obs.shape}. "
            "Expected an array-like observation per timestep."
        )

    # Minari often stores T+1 observations (including final next_obs) for each episode.
    if obs.shape[0] == num_steps + 1:
        obs = obs[:-1]
    elif obs.shape[0] != num_steps:
        raise ValueError(
            f"Observation length {obs.shape[0]} does not match number of steps {num_steps}."
        )

    if acts.shape[0] != num_steps:
        raise ValueError(
            f"Action length {acts.shape[0]} does not match number of steps {num_steps}."
        )

    path = {
        "observations": obs,
        "actions": acts,
        "rewards": rewards,
        "terminals": terminations,
        # store truncations as timeouts so downstream code can treat them as episode ends
        "timeouts": truncations,
    }
    return path


def download_d4rl_data() -> None:
    """Download D4RL-style datasets via Minari and convert to .pkl trajectories.

    The output .pkl files are saved under data/d4rl_dataset/ with names matching
    the original D4RL env IDs (e.g. antmaze-medium-diverse-v2.pkl), so the rest
    of the RF code can load them unchanged.
    """
    data_dir = "data/d4rl_dataset/"
    os.makedirs(data_dir, exist_ok=True)
    print(f"Saving trajectories to: {data_dir}")

    # Mapping from the D4RL environment IDs used elsewhere in this repo
    # to the corresponding Minari dataset IDs.
    d4rl_to_minari = {
        "antmaze-medium-diverse-v2": "D4RL/antmaze/medium-diverse-v1",
        "hopper-medium-v2": "mujoco/hopper/medium-v0",
        # Hopper Simple dataset (no direct D4RL counterpart).
        # We treat "hopper-simple-v0" as a D4RL-style ID used only inside
        # this codebase and save the converted trajectories under that name.
        "hopper-simple-v0": "mujoco/hopper/simple-v0",
        # Additional Mujoco tasks.
        "walker2d-medium-v2": "mujoco/walker2d/medium-v0",
        # Maze and Kitchen tasks from the D4RL suite.
        "maze2d-medium-v1": "D4RL/pointmaze/medium-v2",
        "kitchen-mixed-v0": "D4RL/kitchen/mixed-v2",
        # If you want more AntMaze variants, add them here, e.g.:
        # "antmaze-medium-play-v2": "D4RL/antmaze/medium-play-v1",
        # "antmaze-umaze-v2": "D4RL/antmaze/umaze-v1",
        # "antmaze-umaze-diverse-v2": "D4RL/antmaze/umaze-diverse-v1",
        # "antmaze-large-play-v2": "D4RL/antmaze/large-play-v1",
        # "antmaze-large-diverse-v2": "D4RL/antmaze/large-diverse-v1",
    }

    for d4rl_env_id, minari_id in d4rl_to_minari.items():
        pkl_file_path = os.path.join(data_dir, d4rl_env_id)

        print(f"Processing D4RL dataset: {d4rl_env_id} (Minari ID: {minari_id})")

        try:
            # This requires minari[hf] and internet access the first time.
            minari.download_dataset(minari_id)
        except Exception as exc:
            print(
                f"\tFailed to download {minari_id} from Minari.\n"
                f"\tMake sure `huggingface_hub` is installed "
                f"(e.g. `pip install \"minari[hf]\"`) and you have network access.\n"
                f"\tError: {exc}"
            )
            continue

        dataset = minari.load_dataset(minari_id)
        paths: List[Dict[str, np.ndarray]] = [
            _episode_to_path(ep) for ep in dataset.iterate_episodes()
        ]

        if not paths:
            print(f"\tNo episodes found for {minari_id}, skipping.")
            continue

        returns = np.array([traj["rewards"].sum() for traj in paths])
        num_samples = int(np.sum([traj["rewards"].shape[0] for traj in paths]))

        print(f"\tNumber of samples collected: {num_samples}")
        print(
            "\tTrajectory returns: "
            f"mean = {returns.mean():.4f}, "
            f"std = {returns.std():.4f}, "
            f"max = {returns.max():.4f}, "
            f"min = {returns.min():.4f}"
        )

        with open(f"{pkl_file_path}.pkl", "wb") as f:
            pickle.dump(paths, f)
        print(f"\tSaved trajectories to {pkl_file_path}.pkl")


if __name__ == "__main__":
    download_d4rl_data()
