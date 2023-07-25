#!/bin/bash

#SBATCH -N 1
#SBATCH -c 8
#SBATCH -G 1
#SBATCH -p comm_gpu_inter

cd $SLURM_SUBMIT_DIR

PYTORCH_IMAGE=/shared/containers/NGC_PyTorch_23.06-py3.sif
PYTHON_SCRIPT=example_pytorch.py

nvidia-smi

module load singularity

singularity exec --nv $PYTORCH_IMAGE python3 $PYTHON_SCRIPT

