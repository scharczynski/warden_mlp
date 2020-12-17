#!/bin/bash -l
#$ -l h_rt=500:00:00
#$ -P ecog-eeg
#$ -N run_all_first

module load python3

 qsub arrayjob_all.sh 1 425 ./warden_recall_first_main_MT.py
 qsub arrayjob_all.sh 1 442 ./warden_recog_first_main_MT.py
 qsub arrayjob_all.sh 1 425 ./warden_recall_first_stim_MT.py
 qsub arrayjob_all.sh 1 442 ./warden_recog_first_stim_MT.py