#!/bin/sh
#BSUB -q gpua100
#BSUB -J task12
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o BatchJobOutput/ProjectGPU_task12%J.out
#BSUB -e BatchJobOutput/ProjectGPU_task12%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python task12.py 4571