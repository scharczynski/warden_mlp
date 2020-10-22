#!/bin/bash -
#$ -l h_rt=100:00:00
#$ -P ecog-eeg
#$ -N run_all_cells

module load python3

qsub -t $1-$2 arrayjob_one.sh $3



    