I will outline here the steps required to run and use this code:
Any instructions in tickmarks (`` ``) should be run in the terminal as written.
Depending on your Python install and setup commands like ``python3`` or ``pip3`` could be ``python`` or ``pip``. 

Let me know if any of these steps doesn't work.

Prerequisites:
     We need to have the fitting package "maxlikespy" downloaded onto your cluster user and installed. 

        1. Choose where you want to install the package.
        2. ``git clone https://github.com/tcnlab/maxlikespy.git``
        3. ``module load python3``
        4. cd into the cloned directory
        5. ``pip3 install --user .``
   
    We also want the package "autograd" for the solver:
        1. "pip3 install --user autograd"

    Now we want to clone the "project" directory I've made
        1. cd where you want to store these files (anywhere as long as not inside maxlikespy)
        2. ``git clone https://github.com/scharczynski/warden_mlp.git``

Usage:
    Running the code requires only one job command from inside the project folder.
    Currently I have 4 run scripts corresponding to either first or both presentations and recog/recall trials:
        
    ``qsub run_all_cells.sh 0 442 ./warden_recog_trials.py``
    ``qsub run_all_cells.sh 0 442 ./warden_recog_both.py``
    or 

    ``qsub run_all_cells.sh 0 425 ./warden_recall_trials.py``
    ``qsub run_all_cells.sh 0 425 ./warden_recall_both.py``
 
    This tells the cluster to submit the job script "run_all_cells.sh" and submit the units in the range shown, using the run script.

    The run script should create your chosen save directory if it doesn't exist, but creating it by hand isn't a bad idea.

Cleanup:
    maxlikespy generates a whole mess of output files in order to keep everything as parallel as possible.
    I have written a script that transforms all output into 3 bigger files and downloads them to your machine.
    This step is of course optional but I find it very helpful. 

    This script is included in the git repository but it is meant to be run locally on your machine.
    One simple way to get it off the cluster is to open terminal on your machine and cd to where you want the script to be and input: ``scp YOUR_USERNAME@scc1.bu.edu:/PATH/TO/YOUR/PROJECT/DIR/maxlikespy_cleanup.py .``

    Once you have this locally, we'll need a few packages:
        ``pip3 install paramiko``
        ``pip3 install scp``

    Inside the script, you must ensure variables:
        user, password, run_path, data_path, output_path, local_path
    are all set according to your setup.

    Once we complete a run on the cluster (needs to be completely finished), we can run this script like:
    ``python3 maxlikespy_cleanup.py warden/recog_trials True False`` 
    or 
    ``python3 maxlikespy_cleanup.py warden/recall_trials True False``


    where the first argument is the project/data, second boolean is whether you ran even odd trials, and last boolean argument is whether you want to download the premade raster/fit plots.

    The main caveat with using this script as written is you must have your save directory, local directory named the same as the directory the data is stored in. In this case the data is in "warden/recog_trials" or "warden/recall_trials". 

    This script is obviously not neccesary to use, but if you want to handle the output on your machine it's the easiest way to do so.
    
