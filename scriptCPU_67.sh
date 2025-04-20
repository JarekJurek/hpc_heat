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

source /zhome/69/0/168594/Documents/HPC_course/venv/bin/activate

python task7.py numba dynamic