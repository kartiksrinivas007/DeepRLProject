
# OpenAI Gym

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are shared with the main Reinformer codebase and are stored under `../data/d4rl_dataset/` (relative to this folder).
They are downloaded and converted from Minari using:

```bash
cd ..
python data/download_d4rl_datasets.py
cd gym-DT
```

## Example usage

Experiments can be reproduced with the following (from this directory):

```bash
python experiment.py --env hopper --dataset medium --model_type dt --dataset_dir ../data/d4rl_dataset/
```

Adding `-w True` will log results to Weights and Biases.
