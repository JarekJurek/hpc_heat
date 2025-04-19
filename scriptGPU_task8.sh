#!/bin/sh
#BSUB -q c02613
#BSUB -J task8
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:10
#BSUB -o BatchJobOutput/ProjectGPU_task8%J.out
#BSUB -e BatchJobOutput/ProjectGPU_task8%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python task8.py 10