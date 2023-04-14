#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output="%A\_%x.txt"
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu
#SBATCH --job-name=resnet

module purge

singularity exec --nv --overlay /scratch/jsr10000/pytorch-example/my_pytorch.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python3 main.py"




