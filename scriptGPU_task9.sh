#!/bin/sh
#BSUB -q c02613
#BSUB -J task9
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:10
#BSUB -o BatchJobOutput/ProjectGPU_task9%J.out
#BSUB -e BatchJobOutput/ProjectGPU_task9%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python task9.py 10