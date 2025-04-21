#!/bin/bash
#BSUB -J task67
#BSUB -q hpc
#BSUB -W 3:00
#BSUB -R "rusage[mem=1024MB]"
#BSUB -o BatchJobOutput/ProjectCPU_task67_%J.out
#BSUB -e BatchJobOutput/ProjectCPU_task67_%J.err

#BSUB -R "select[model == XeonGold6126]"

#BSUB -R "span[hosts=1]"
#BSUB -n 20

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# For task 6
# python task6_and_7.py dynamic

# For task 7
# python task6_and_7.py numba

# To run both task 6 and 7 together:
python task6_and_7.py numba dynamic