# DeepRLProject Overview

This repository contains several related offline RL codebases:

- **Rein*for*mer** (RF/) – ICML 2024 sequence‑modeling for offline RL.
- **Homework 6 TD3** (RF/hw6_starter_code/) – TD3 + offline buffer on Minari/D4RL tasks.
- **Decision Transformer baseline** (RF/gym-DT/) – DT and BC experiments in Gym environments.

The shared environment is defined in `env.yml` at the repo root.

---

## Environment Setup

From the repository root:

```bash
conda env create -f env.yml   # creates env named "703"
conda activate 703
```

### Mujoco / mujoco-py

The environment depends on `mujoco` and `mujoco-py`. You still need to:

1. Download the Mujoco binary (e.g. MuJoCo 2.x) and place it in the expected directory (such as `~/.mujoco/mujoco210`), and
2. Set the appropriate environment variables (`MUJOCO_PY_MUJOCO_PATH`, `LD_LIBRARY_PATH`, etc.)

Follow the official instructions here exactly:

https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco

Once Mujoco is installed and the environment is active, you can run all experiments below.

---

## Repository Structure

```text
DeepRLProject/
  env.yml                  # Conda environment (name: 703)
  envs/                    # Misc. course/utility code
  main.py                  # Top-level script (course-specific)
  search/, torch/, utils.py

  RF/                      # Reinformer + offline RL experiments
    README.md              # RF-specific README (details commands within RF/)
    data/
      d4rl_dataset/        # Offline trajectories (D4RL-style .pkl)
      buffer_cache/        # Cached replay buffers for TD3 homework
      download_d4rl_datasets.py

    hw6_starter_code/      # Homework 6 TD3 offline RL
      runner.py            # TD3 training loop (Minari eval + D4RL buffers)
      utils.py             # Buffer loading, plotting, eval
      td3_agent.py         # TD3 + BC + CQL agent
      buffer.py, policies.py
      sbatch_scripts/
        logs/
        td3_hopper_bc100.sbatch
        td3_hopper_bc1000.sbatch
        td3_hopper_cql5.sbatch
        td3_hopper_cql5_seed2.sbatch
        td3_hopper_buffers.sh

    main.py                # Reinformer training entry point
    eval.py, trainer.py, model/
    scripts/               # Simple shell wrappers for Reinformer runs
      hopper_command.sh
      maze_command.sh
      walker_command.sh
      kitchen_command.sh
    sbatch_scripts/        # SLURM jobs for Reinformer
      hopper_reinformer.sbatch
      maze2d_reinformer.sbatch
      walker2d_reinformer.sbatch
      kitchen_reinformer.sbatch
      submit_all_reinformer.sh
      logs/

    gym-DT/                # Decision Transformer + BC baseline in Gym
      conda_env.yml        # Original DT env (use root env.yml instead here)
      experiment.py        # DT / BC training script
      decision_transformer/
      scripts/
        run_dt_hopper.sh
      sbatch_scripts/
        bc_hopper_medium.sbatch
        dt_hopper_medium.sbatch
        logs/
```

---

## Shared Data (RF/data)

All three components (Reinformer, Homework 6 TD3, and gym-DT) use the same D4RL‑style offline datasets.

From `RF/` run:

```bash
cd RF
python data/download_d4rl_datasets.py
```

This uses Minari to download datasets once and converts them into `.pkl` trajectories under `RF/data/d4rl_dataset/`.

---

## Homework 6 TD3 (RF/hw6_starter_code)

### Direct run

From `RF/hw6_starter_code/`:

```bash
conda activate 703

python runner.py \
  --agent td3 \
  --env_id mujoco/hopper/medium-v0 \
  --total_steps 100000
```

The runner:

- Loads offline trajectories from `RF/data/d4rl_dataset/*.pkl` into a replay `Buffer` (with on‑disk caching in `RF/data/buffer_cache/`).
- Recovers the evaluation environment via Minari (`minari.load_dataset(...).recover_environment(eval_env=True)`), falling back to a Gym env if needed.
- Logs raw and normalized evaluation scores, plus TD3/BC/CQL losses.

### SLURM sbatch jobs (TD3 variants)

From `RF/hw6_starter_code/sbatch_scripts/`:

- Submit three Hopper TD3 variants:

```bash
bash td3_hopper_buffers.sh
```

- Or submit individually:

```bash
# TD3 + BC regularization, weight 100
sbatch td3_hopper_bc100.sbatch

# TD3 + BC regularization, weight 1000
sbatch td3_hopper_bc1000.sbatch

# TD3 + CQL (alpha = 5)
sbatch td3_hopper_cql5.sbatch

# TD3 + CQL (alpha = 5) with different seed
sbatch td3_hopper_cql5_seed2.sbatch
```

Logs: `RF/hw6_starter_code/sbatch_scripts/logs/`.

---

## Reinformer Experiments (RF/main.py)

### Quick start

After preparing data:

```bash
cd RF
conda activate 703

python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/
```

This trains the Reinformer model on Hopper‑medium, using:

- Offline trajectories from `data/d4rl_dataset/`,
- Minari/D4RL to build evaluation environments and normalized scores,
- Evaluation curves saved under `RF/plots/`.

### Convenience shell scripts

From `RF/`:

```bash
# Hopper medium
bash scripts/hopper_command.sh

# Maze2d medium
bash scripts/maze_command.sh

# Walker2d medium
bash scripts/walker_command.sh

# Kitchen mixed
bash scripts/kitchen_command.sh
```

Each script calls `python main.py` with sensible defaults for that environment/dataset.

### SLURM sbatch jobs

From `RF/sbatch_scripts/`:

```bash
# Submit all Reinformer runs
bash submit_all_reinformer.sh

# Or individual jobs
sbatch hopper_reinformer.sbatch
sbatch maze2d_reinformer.sbatch
sbatch walker2d_reinformer.sbatch
sbatch kitchen_reinformer.sbatch
```

Logs: `RF/sbatch_scripts/logs/`.

---

## Decision Transformer (RF/gym-DT)

The `gym-DT/` folder contains the Decision Transformer / BC baseline experiments adapted from the original DT code.

### Direct run

From `RF/gym-DT/`:

```bash
conda activate 703

python experiment.py \
  --env hopper \
  --dataset medium
```

or use the helper script:

```bash
bash scripts/run_dt_hopper.sh
```

Evaluation plots are saved in `RF/gym-DT/plots/` (e.g. normalized score curves for various target returns).

### SLURM sbatch jobs

From `RF/gym-DT/sbatch_scripts/`:

```bash
# Behavior Cloning on Hopper-medium
sbatch bc_hopper_medium.sbatch

# Decision Transformer on Hopper-medium
sbatch dt_hopper_medium.sbatch
```

Logs: `RF/gym-DT/sbatch_scripts/logs/`.

---

## Where to Look Next

- For detailed hyperparameters and options, inspect:
  - `RF/hw6_starter_code/runner.py` (TD3/BC/CQL flags),
  - `RF/main.py` (Reinformer options),
  - `RF/gym-DT/experiment.py` (DT/BC options).
- For environment issues (Mujoco rendering, D4RL warnings), revisit the Mujoco‑py installation guide and confirm your environment variables and GPU drivers are configured correctly.

