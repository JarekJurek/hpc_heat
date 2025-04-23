#!/bin/sh
#BSUB -q c02613
#BSUB -J task10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:10
#BSUB -o BatchJobOutput/ProjectGPU_task10%J.out
#BSUB -e BatchJobOutput/ProjectGPU_task10%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# nsys profile -o task10_prof python task9.py 10
nsys profile -o task10_prof_fixed python task10.py 10