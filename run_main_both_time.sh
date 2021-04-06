#!/bin/bash -l
#$ -l h_rt=500:00:00
#$ -P ecog-eeg
#$ -N run_all_first

module load python3

 qsub arrayjob_all.sh 1 425 ./warden_recall_both_time_main.py
 qsub arrayjob_all.sh 1 442 ./warden_recog_both_time_main.py
#  qsub arrayjob_all.sh 1 425 ./warden_recall_both_time_stim.py
#  qsub arrayjob_all.sh 1 442 ./warden_recog_both_time_stim.py