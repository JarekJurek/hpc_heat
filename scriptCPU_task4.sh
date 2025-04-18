#!/bin/bash
#BSUB -J ProjectCPU
#BSUB -q hpc
#BSUB -W 1:00
#BSUB -R "rusage[mem=1024MB]"
#BSUB -o BatchJobOutput/ProjectCPU_%J.out
#BSUB -e BatchJobOutput/ProjectCPU_%J.err

#BSUB -R "select[model == XeonGold6126]"

#BSUB -R "span[hosts=1]"
#BSUB -n 4

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

kernprof -l simulate.py 10 


### After that:
### python -m line_profiler simulate.py.lprof > prof_initial_jacobi.txt
