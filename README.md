# BCCR Training Programme in Machine Learning
---
_Setup your ML workflow_


This repository is an example machine learning project.


## Reproducible Environment (macOS, Linux, Windows)

This project uses a two-step setup:

1. Create a reproducible base environment from `extras/environment.yml`.
2. Install the platform-appropriate PyTorch backend (CUDA, MPS, or CPU).

### 1) Create or update base env

```bash
conda env create -f extras/environment.yml || conda env update -f extras/environment.yml --prune
```

### 2) Install PyTorch backend

```bash
bash scripts/setup_pytorch_backend.sh --backend auto
```

Optional explicit backend selection:

```bash
bash scripts/setup_pytorch_backend.sh --backend cuda --cuda-version 12.1
bash scripts/setup_pytorch_backend.sh --backend mps
bash scripts/setup_pytorch_backend.sh --backend cpu
```

### 3) Verify backend

```bash
conda run -n bccr-ml-project python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"
```

### Fully reproducible lock files (recommended for team usage)

Install conda-lock and generate platform-specific lock files:

```bash
pip install conda-lock
conda-lock -f extras/environment.yml -p osx-arm64 -p osx-64 -p linux-64 -p win-64
```

Then create environments from lock files on each platform for maximum reproducibility.

### Setup weights and biases

Login to weights and biases to make sure experiments will get logged properly to your account.

```bash
wandb login
```


