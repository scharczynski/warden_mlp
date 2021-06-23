#!/bin/bash -l
#$ -l h_rt=500:00:00
#$ -P ecog-eeg
#$ -N cell

module load python3
cell=$(($SGE_TASK_ID - 1))
python3 $1 $cell $cell