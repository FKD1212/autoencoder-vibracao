#!/bin/bash
# Launcher do projeto no WSL2 com GPU habilitada.
# Uso: bash run_wsl.sh          (roda main.py)
#      bash run_wsl.sh script.py (roda outro script)

SITE=~/tf_gpu/lib/python3.12/site-packages
export LD_LIBRARY_PATH=/usr/lib/wsl/lib\
:$SITE/nvidia/cublas/lib\
:$SITE/nvidia/cuda_runtime/lib\
:$SITE/nvidia/cudnn/lib\
:$SITE/nvidia/cufft/lib\
:$SITE/nvidia/curand/lib\
:$SITE/nvidia/cusolver/lib\
:$SITE/nvidia/cusparse/lib\
:$SITE/nvidia/nccl/lib\
:$SITE/nvidia/nvjitlink/lib\
${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

SCRIPT="${1:-/mnt/d/Felipe Pasta/Autoencoder/main.py}"
exec ~/tf_gpu/bin/python3 "$SCRIPT"
