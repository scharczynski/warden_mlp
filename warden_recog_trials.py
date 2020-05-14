import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting
import json

# def run_script(cell_range):
    # # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # # save_dir = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recog_trials/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/"

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 1500])
    # solver_params = {
    #     "niter": 1000,
    #     "stepsize": 100,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # bounds_smtstim = {
    #     "sigma": [0, 1000.],
    #     "mu": [0, 1500.],
    #     "tau": [20, 20000.],
    #     "a_1": [10**-10, 1/5.],
    #     "a_2": [10**-10, 1/5.],
    #     "a_3": [10**-10, 1/5.],
    #     "a_4": [10**-10, 1/5.],
    #     "a_0": [10**-10, 1/5.]
    # }
    # bounds_smt = {
    #     "sigma": [0, 1000.],
    #     "mu": [0, 1500.],
    #     "tau": [20, 20000.],
    #     "a_1": [10**-10, 1/2.],
    #     "a_0": [10**-10, 1/2.]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "Const","SigmaMuTau", "SigmaMuTauStim"], save_dir=save_dir)
    # pipeline.set_model_bounds("SigmaMuTau", bounds_smt)
    # pipeline.set_model_bounds("SigmaMuTauStim", bounds_smtstim)
    # pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # # with open("/Users/stevecharczynski/workspace/data/warden/recog_trials/info.json") as f:
    # with open("/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/info.json") as f:
    #     stims = json.load(f)
    #     stims = {int(k):v for k,v in stims.items()}
    # pipeline.set_model_info("SigmaMuTauStim", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_x0("SigmaMuTauStim", [0.01, 1000, 100, 1e-1, 1e-1,1e-1, 1e-1, 1e-1])
    # pipeline.set_model_x0("SigmaMuTau", [0.01, 1000, 100, 1e-1, 1e-1])
    # pipeline.set_model_x0("Const", [1e-1])
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    # pipeline.compare_even_odd("SigmaMuTau", "SigmaMuTauStim", 0.01)
    # pipeline.compare_models("Const", "SigmaMuTau", 0.01, smoother_value=100)
    # pipeline.compare_models("SigmaMuTau", "SigmaMuTauStim", 0.01, smoother_value=100)

def run_script(cell_range):
    # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # save_dir = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recog_trials/"
    path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/"

    # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    data_processor = analysis.DataProcessor(
        path_to_data, cell_range, window=[0, 1500])
    solver_params = {
        "niter": 1000,
        "stepsize": 1000,
        "interval": 20,
        "method": "TNC",
        "use_jac": True,
        "T" : 1,
        "disp":False
    }
    bounds_smtstim = {
        "sigma": [0, 1000.],
        "mu": [0, 1500.],
        "tau": [20, 2000.],
        "a_1": [0., 1/5.],
        "a_2": [0., 1/5.],
        "a_3": [0., 1/5.],
        "a_4": [0., 1/5.],
        "a_0": [0., 1/5.]
    }
    bounds_smt = {
        "sigma": [0, 1000.],
        "mu": [0, 1500.],
        "tau": [20, 2000.],
        "a_1": [0., 1/2.],
        "a_0": [0., 1/2.]
    }
    pipeline = analysis.Pipeline(cell_range, data_processor, [
        "Const","SigmaMuTau", "SigmaMuTauStimWarden"], save_dir=save_dir)
    pipeline.set_model_bounds("SigmaMuTau", bounds_smt)
    pipeline.set_model_bounds("SigmaMuTauStimWarden", bounds_smtstim)
    pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # with open("/Users/stevecharczynski/workspace/data/warden/recog_trials/info.json") as f:
    with open("/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/info.json") as f:
        stims = json.load(f)
        stims = {int(k):v for k,v in stims.items()}
    pipeline.set_model_info("SigmaMuTauStimWarden", "stim_identity", stims, per_cell=True)
    pipeline.set_model_x0("SigmaMuTauStimWarden", [10, 1000, 100, 1e-1, 1e-1,1e-1, 1e-1, 1e-1])
    pipeline.set_model_x0("SigmaMuTau", [10, 1000, 100, 1e-1, 1e-1])
    pipeline.set_model_x0("Const", [1e-1])
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    # pipeline.compare_even_odd("SigmaMuTau", "SigmaMuTauStimWarden", 0.01)
    pipeline.compare_models("Const", "SigmaMuTau", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "SigmaMuTauStimWarden", 0.01, smoother_value=100)
    pipeline.compare_models("SigmaMuTau", "SigmaMuTauStimWarden", 0.01, smoother_value=100)
    pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    pipeline.compare_even_odd("Const", "SigmaMuTauStimWarden", 0.01)
    pipeline.compare_even_odd("SigmaMuTau", "SigmaMuTauStimWarden", 0.01)


    # # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # # save_dir = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recog_trials/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/"

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 1500])
    # solver_params = {
    #     "niter": 3000,
    #     "stepsize": 100,
    #     "interval": 10,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # bounds_smt = {
    #     "sigma": [0, 1000.],
    #     "mu": [0, 1500.],
    #     "tau": [20, 20000.],
    #     "a_1": [10**-10, 1/2.],
    #     "a_0": [10**-10, 1/2.]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "Const","SigmaMuTau"], save_dir=save_dir)
    # pipeline.set_model_bounds("SigmaMuTau", bounds_smt)
    # pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # # with open("/Users/stevecharczynski/workspace/data/warden/recog_trials/info.json") as f:
    # pipeline.set_model_x0("SigmaMuTau", [0.01, 1000, 100, 1e-1, 1e-1])
    # pipeline.set_model_x0("Const", [1e-1])
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    # pipeline.compare_models("Const", "SigmaMuTau", 0.01, smoother_value=100)

if __name__ == "__main__":
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range)
