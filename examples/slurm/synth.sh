#!/bin/bash

#SBATCH -p bigmem
#SBATCH --job-name=synth_job_array
#SBATCH --output=synth_job_array.%A_%a.out
#SBATCH --error=synth_job_array.%A_%a.err
#SBATCH -t 8:00:00
#SBATCH --mem=800GB
#SBATCH -c 1
#SBATCH --array=1-4


RANK_PARAM=$SLURM_ARRAY_TASK_ID

hostname
srun hostname
cd /home/users/tetianap/multilevel_factor_model/examples/slurm
source /home/groups/boyd/tetianap/anaconda3/bin/activate mlr

python -u synth.py --slurm $RANK_PARAM
