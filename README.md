# ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
## 
ForceVLA is based on the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based diffusion vision-language-action model (VLA)； Both training and inference are based on π₀.

## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.


## dataset
https://huggingface.co/datasets/qiaojunyu/ForceVLA-real-data

## Installation

When cloning this repo, make sure to update submodules:

```bash
conda create -n forcevla python=3.11 -y

```

```bash
python -m pip install --upgrade pip setuptools wheel
conda install -c nvidia cuda-toolkit=12.8
```

```bash
cd lerobot/
conda install ffmpeg=7.1.1 -c conda-forge
pip install -e .
```

```bash
cd ./ForceVLA
pip install -e .
```

```bash
cd dlimp/
pip install -e .
```

```bash
cd packages/
cd openpi-client/
pip install -e .
```

```bash
cd flaxformer/
pip install -e .
```
## train policy
```bash
export HF_LEROBOT_HOME="xxxxxx"
python scripts/compute_norm_stats.py --config-name forcevla_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  python scripts/train.py forcevla_lora --exp-name=my_experiment --overwrite
```
