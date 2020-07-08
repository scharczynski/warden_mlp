import os
from scp import SCPClient
import sys
import paramiko
import datetime
import time


server = "scc1.bu.edu"
user = "scharcz" #cluster username
password =  #cluster password. don't make this file public! 
run_path = "/projectnb/ecog-eeg/stevechar/cluster_files/" #path to your project directory 
# run_path = "/projectnb/ecog-eeg/stevechar/projects/warden_mlp/"
# run_path = "/projectnb/ecog-eeg/stevechar/ml_runs/jay/lec/"

timestr = time.strftime("%Y%m%d-%H%M%S")
experiment = sys.argv[1]
even_odd = sys.argv[2]
download_figs = sys.argv[3]
data_path = "/projectnb/ecog-eeg/stevechar/data/{0}".format(experiment) #path to where data is stored
output_path = "/projectnb/ecog-eeg/stevechar/ml_runs/{0}".format(experiment) #the save directory you chose
local_path = "/Users/stevecharczynski/workspace/data/{0}".format(experiment) #the directory you are planning on saving output to locally
run_date =  datetime.date.today()
local_full_path = "{0}/results/{1}".format(local_path, timestr)
os.makedirs(local_full_path, exist_ok=True)

def output_pwd():
    stdin, stdout, stderr = ssh.exec_command("pwd")
    print(stdout.read())

def createSSHClient(server, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, username=user, password=password)
    return client

ssh = createSSHClient(server, user, password)
scp = SCPClient(ssh.get_transport())
stdin, stdout, stderr = ssh.exec_command("cd {0}/spikes;ls -1 | wc -l".format(data_path))
num_cells = int(stdout.read())

stdin, stdout, stderr = ssh.exec_command("module load python3")
print("Stitching output files, found {0} cells".format(num_cells))

stdin, stdout, stderr = ssh.exec_command("cd {0}; python3 stitch_output.py {1} {2} {3}".format(run_path, output_path, "0", str(num_cells-1)))

if even_odd == "True":
    stdin, stdout, stderr = ssh.exec_command("cd {0}; python3 stitch_output_evenodd.py {1} {2} {3}".format(run_path, output_path, "0", str(num_cells-1)))
    print(stderr.read(), stdout.read())
    scp.get("{0}/results/log_likelihoods_even.json".format(output_path), local_full_path)
    scp.get("{0}/results/model_comparisons_even.json".format(output_path), local_full_path)
    scp.get("{0}/results/cell_fits_even.json".format(output_path), local_full_path)
    scp.get("{0}/results/log_likelihoods_odd.json".format(output_path), local_full_path)
    scp.get("{0}/results/model_comparisons_odd.json".format(output_path), local_full_path)
    scp.get("{0}/results/cell_fits_odd.json".format(output_path), local_full_path)

scp.get("{0}/results/log_likelihoods.json".format(output_path), local_full_path)
scp.get("{0}/results/model_comparisons.json".format(output_path), local_full_path)
scp.get("{0}/results/cell_fits.json".format(output_path), local_full_path)

if download_figs == "True":
    print("Downloading figures")
    scp.get("{0}/results/figs/".format(output_path), local_full_path, True)

print("Cleaning up output files")
stdin, stdout, stderr = ssh.exec_command("cd {0}/results; mkdir {1}; mv ./*.json ./figs {1}".format(output_path, timestr))
print("Files now in {0}".format(timestr))


