#!/bin/bash -l
#$ -l h_rt=100:00:00
#$ -P ecog-eeg
#$ -N cell_$SGE_TASK_ID_$1

module load python3

python3 $1 $SGE_TASK_ID $SGE_TASK_ID 