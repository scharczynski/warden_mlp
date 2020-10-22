#!/bin/bash -l
#$ -l h_rt=500:00:00
#$ -P ecog-eeg
#$ -N array_all

module load python3

qsub -t $1-$2 arrayjob_one.sh $3



    