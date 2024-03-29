#!/bin/bash -l
#$ -l h_rt=500:00:00
#$ -P ecog-eeg
#$ -N run_all_first

module load python3

 qsub arrayjob_all.sh 1 425 ./warden_recall_second_main_time.py
 qsub arrayjob_all.sh 1 442 ./warden_recog_second_main_time.py
 qsub arrayjob_all.sh 1 425 ./warden_recall_second_stim_time.py
 qsub arrayjob_all.sh 1 442 ./warden_recog_second_stim_time.py