# Rein*for*mer & Homework 6 Layout

Official code for ICML 2024 paper **Rein*for*mer**: Max-Return Sequence Modeling for offline RL, plus additional Homework 6 and Decision Transformer code.

Here is the overview of our proposed **Rein*for*mer**. For more details, please refer to our paper https://arxiv.org/pdf/2405.08740.
![overview](https://github.com/Dragon-Zhuang/Reinformer/assets/47406013/698651f2-24c5-4734-9423-97088058bef7)

---

## Repository Structure (RF/)

From the project root you are in `RF/`:

```text
RF/
  data/
    d4rl_dataset/          # Offline trajectories (.pkl) used by all components
    download_d4rl_datasets.py

  hw6_starter_code/        # Homework 6 TD3 + offline RL
    runner.py              # Main TD3 training loop
    utils.py               # Buffer loading, eval, plotting
    td3_agent.py
    sbatch_scripts/
      logs/
      td3_hopper_bc100.sbatch
      td3_hopper_bc1000.sbatch
      td3_hopper_cql5.sbatch
      td3_hopper_cql5_seed2.sbatch
      td3_hopper_buffers.sh

  # Original Reinformer codebase
  main.py                  # Reinformer training entry point
  eval.py
  model/, trainer.py       # Reinformer model + trainer
  scripts/
    hopper_command.sh
    maze_command.sh
    walker_command.sh
    kitchen_command.sh
  sbatch_scripts/
    hopper_reinformer.sbatch
    maze2d_reinformer.sbatch
    walker2d_reinformer.sbatch
    kitchen_reinformer.sbatch
    submit_all_reinformer.sh
    logs/

  # Gym-Decision Transformer baseline
  gym-DT/
    experiment.py          # DT/Reinformer experiments in Gym
    decision_transformer/
    scripts/
      run_dt_hopper.sh
    sbatch_scripts/
      bc_hopper_medium.sbatch
      dt_hopper_medium.sbatch
      logs/
```

---

## Data Preparation

All components share the same offline trajectories under `data/d4rl_dataset/`.

From `RF/`:

```bash
python data/download_d4rl_datasets.py
```

This uses Minari to download and convert datasets once into `.pkl` files.

---

## Homework 6 TD3 (hw6_starter_code)

### Direct run

From `RF/hw6_starter_code/`:

```bash
python runner.py \
  --agent td3 \
  --env_id mujoco/hopper/medium-v0 \
  --total_steps 100000
```

### SLURM sbatch scripts

From `RF/hw6_starter_code/sbatch_scripts/`:

- Run three Hopper TD3 variants (BC weights and CQL) at once:

  ```bash
  bash td3_hopper_buffers.sh
  ```

- Individually:

  ```bash
  # BC regularization weight 100
  sbatch td3_hopper_bc100.sbatch

  # BC regularization weight 1000
  sbatch td3_hopper_bc1000.sbatch

  # CQL alpha = 5
  sbatch td3_hopper_cql5.sbatch

  # CQL alpha = 5 with a different seed
  sbatch td3_hopper_cql5_seed2.sbatch
  ```

Logs go to `RF/hw6_starter_code/sbatch_scripts/logs/`.

---

## Reinformer (RF/main.py)

### Quick start (single-GPU run)

From `RF/`:

```bash
python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/
```

### Convenience shell scripts

From `RF/`:

```bash
# Hopper medium Reinformer run
bash scripts/hopper_command.sh

# Maze2d medium
bash scripts/maze_command.sh

# Walker2d medium
bash scripts/walker_command.sh

# Kitchen mixed
bash scripts/kitchen_command.sh
```

### SLURM sbatch scripts

From `RF/sbatch_scripts/`:

```bash
# Submit all Reinformer jobs
bash submit_all_reinformer.sh

# Or individually
sbatch hopper_reinformer.sbatch
sbatch maze2d_reinformer.sbatch
sbatch walker2d_reinformer.sbatch
sbatch kitchen_reinformer.sbatch
```

Logs are written under `RF/sbatch_scripts/logs/`.

---

## Gym-Decision Transformer (gym-DT)

### Direct run

From `RF/gym-DT/`:

```bash
python experiment.py \
  --env hopper \
  --dataset medium
```

or use the helper script:

```bash
bash scripts/run_dt_hopper.sh
```

### SLURM sbatch scripts

From `RF/gym-DT/sbatch_scripts/`:

```bash
# Behavior Cloning baseline on Hopper-medium
sbatch bc_hopper_medium.sbatch

# Decision Transformer on Hopper-medium
sbatch dt_hopper_medium.sbatch
```

Logs are written under `RF/gym-DT/sbatch_scripts/logs/`.  This mirrors the same dataset directory (`data/d4rl_dataset/`) as the Reinformer and Homework 6 code.  Use these scripts as templates for additional environments or datasets you may want to add.  Each component (Reinformer, hw6 TD3, and Gym-DT) can be launched independently but shares the same offline data.  Refer to the respective `runner.py` / `main.py` / `experiment.py` files for more advanced configuration options.  For example, you can change evaluation frequency, batch sizes, or context lengths directly via command-line flags.  This README only lists the most commonly used commands.  For detailed hyperparameter descriptions, inspect the `argparse` definitions near the bottoms of those files.
