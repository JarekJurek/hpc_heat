#!/bin/bash
#BSUB -J task5
#BSUB -q hpc
#BSUB -W 1:00
#BSUB -R "rusage[mem=1024MB]"
#BSUB -o BatchJobOutput/ProjectCPU_task5_%J.out
#BSUB -e BatchJobOutput/ProjectCPU_task5_%J.err

#BSUB -R "select[model == XeonGold6126]"

#BSUB -R "span[hosts=1]"
#BSUB -n 20

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python task5.py 100