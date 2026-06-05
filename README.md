# BCCR Training Programme in Machine Learning
---
_Setup your ML workflow_

This repository is an example machine learning project.

## If running on HubroHub

If you are running this on HubroHub, you need to install lightning and wandb everytime you open a new session:
```bash
pip install wandb lightning
```

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
bash extras/setup_pytorch_backend.sh --backend auto
```

Optional explicit backend selection:

```bash
bash extras/setup_pytorch_backend.sh --backend cuda --cuda-version 12.1
bash extras/setup_pytorch_backend.sh --backend mps
bash extras/setup_pytorch_backend.sh --backend cpu
```

### 3) Activate environment

```bash
conda activate bccr-ml-project
```


### 4) Check torch installation on gpu

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"
```

### 4) Setup weights and biases

Once the environment is activate, login to weights and biases to make sure experiments will get logged properly to your account.
You will need a weights and biases account. Then copy your API key and past it in the terminal when asked.

```bash
wandb login
```


