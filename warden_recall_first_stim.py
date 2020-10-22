import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting
import json

def run_script(cell_range):
     # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # save_dir = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recall_trials_first_stim/"
    path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/"

    # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
 
    # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    data_processor = analysis.DataProcessor(
        path_to_data, cell_range, window=[-500, 1500])
    solver_params = {
        "niter": 1500,
        "stepsize": 100,
        "interval": 20,
        "method": "TNC",
        "use_jac": True,
        "T" : 100,
        "disp":False
    }
    bounds_smtstim = {
        "sigma": [1e-4, 1000.],
        "mu": [0, 1500.],
        "tau": [1e-4, 10000.],
        "a_1": [1e-10, 1/2.],
        "a_2": [1e-10, 1/2.],
        "a_3": [1e-10, 1/2.],
        "a_4": [1e-10, 1/2.],
        "a_0": [1e-10, 1/2.]
    }
    bounds_smt = {
        "sigma": [1e-4, 1000.],
        "mu": [0, 1500.],
        "tau": [1e-4, 10000.],
        "a_1": [1e-10, 1/2.],
        "a_0": [1e-10, 1/2.]
    }
    pipeline = analysis.Pipeline(cell_range, data_processor, [
        "Const","SigmaMuTauStim1", \
            "SigmaMuTauStim2", "SigmaMuTauStim3", "SigmaMuTauStim4"], save_dir=save_dir)
    pipeline.set_model_bounds("SigmaMuTauStim1", bounds_smt)
    pipeline.set_model_bounds("SigmaMuTauStim2", bounds_smt)
    pipeline.set_model_bounds("SigmaMuTauStim3", bounds_smt)
    pipeline.set_model_bounds("SigmaMuTauStim4", bounds_smt)
    pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # with open("/Users/stevecharczynski/workspace/data/warden/recog_trials/info.json") as f:
    with open("/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/info.json") as f:
        stims = json.load(f)
        stims = {int(k):v for k,v in stims.items()}
    pipeline.set_model_info("SigmaMuTauStim1", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("SigmaMuTauStim2", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("SigmaMuTauStim3", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("SigmaMuTauStim4", "stim_identity", stims, per_cell=True)
    pipeline.set_model_x0("SigmaMuTauStim1", [10, 10, 100, 1e-1, 1e-1])
    pipeline.set_model_x0("SigmaMuTauStim2", [10, 10, 100, 1e-1, 1e-1])
    pipeline.set_model_x0("SigmaMuTauStim3", [10, 10, 100, 1e-1, 1e-1])
    pipeline.set_model_x0("SigmaMuTauStim4", [10, 10, 100, 1e-1, 1e-1])
    pipeline.set_model_x0("Const", [1e-1])
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    # pipeline.compare_even_odd("SigmaMuTau", "SigmaMuTauStimWarden", 0.01)
    pipeline.compare_models("Const", "SigmaMuTauStim1", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "SigmaMuTauStim2", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "SigmaMuTauStim3", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "SigmaMuTauStim4", 0.01, smoother_value=100)


    # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    # save_dir = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recall_trials/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/"

    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 1500])
    # solver_params = {
    #     "niter": 2000,
    #     "stepsize": 1000,
    #     "interval": 5,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # bounds_gaussianstim = {
    #     "sigma": [1e-10, 1000.],
    #     "mu": [0, 1500.],
    #     "a_1": [1e-10, 1/5.],
    #     "a_2": [1e-10, 1/5.],
    #     "a_3": [1e-10, 1/5.],
    #     "a_4": [1e-10, 1/5.],
    #     "a_0": [1e-10, 1/5.]
    # }
    # bounds_gaussian = {
    #     "sigma": [1e-10, 1000.],
    #     "mu": [0, 1500.],
    #     "a_1": [1e-10, 1/2.],
    #     "a_0": [1e-10, 1/2.]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "Const","Gaussian", "GaussianStim", "GaussianStim1", \
    #         "GaussianStim2", "GaussianStim3", "GaussianStim4"], save_dir=save_dir)
    # pipeline.set_model_bounds("Gaussian", bounds_gaussian)
    # pipeline.set_model_bounds("GaussianStim1", bounds_gaussian)
    # pipeline.set_model_bounds("GaussianStim2", bounds_gaussian)
    # pipeline.set_model_bounds("GaussianStim3", bounds_gaussian)
    # pipeline.set_model_bounds("GaussianStim4", bounds_gaussian)
    # pipeline.set_model_bounds("GaussianStim", bounds_gaussianstim)
    # pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # # with open("/Users/stevecharczynski/workspace/data/warden/recall_trials/info.json") as f:
    # with open("/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/info.json") as f:
    #     stims = json.load(f)
    #     stims = {int(k):v for k,v in stims.items()}
    # pipeline.set_model_info("GaussianStim", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("GaussianStim1", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("GaussianStim2", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("GaussianStim3", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("GaussianStim4", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_x0("GaussianStim", [10, 1000, 1e-1, 1e-1,1e-1, 1e-1, 1e-1])
    # pipeline.set_model_x0("Gaussian", [10, 1000, 1e-1, 1e-1])
    # pipeline.set_model_x0("GaussianStim1", [10, 1000, 1e-1, 1e-1])
    # pipeline.set_model_x0("GaussianStim2", [10, 1000, 1e-1, 1e-1])
    # pipeline.set_model_x0("GaussianStim3", [10, 1000, 1e-1, 1e-1])
    # pipeline.set_model_x0("GaussianStim4", [10, 1000, 1e-1, 1e-1])
    # pipeline.set_model_x0("Const", [1e-1])
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.fit_even_odd(solver_params=solver_params)
    # # pipeline.compare_even_odd("Const", "Gaussian", 0.01)
    # # pipeline.compare_even_odd("Gaussian", "GaussianStim", 0.01)
    # pipeline.compare_models("Const", "Gaussian", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "GaussianStim", 0.01, smoother_value=100)
    # pipeline.compare_models("Gaussian", "GaussianStim", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "GaussianStim1", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "GaussianStim2", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "GaussianStim3", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "GaussianStim4", 0.01, smoother_value=100)
    # pipeline.compare_models("GaussianStim1", "GaussianStim", 0.01, smoother_value=100)
    # pipeline.compare_models("GaussianStim2", "GaussianStim", 0.01, smoother_value=100)
    # pipeline.compare_models("GaussianStim3", "GaussianStim", 0.01, smoother_value=100)
    # pipeline.compare_models("GaussianStim4", "GaussianStim", 0.01, smoother_value=100)
    # pipeline.compare_even_odd("Const", "Gaussian", 0.01)
    # pipeline.compare_even_odd("Const", "GaussianStim", 0.01)
    # pipeline.compare_even_odd("Gaussian", "GaussianStim", 0.01)
    # pipeline.compare_even_odd("Const", "GaussianStim1", 0.01)
    # pipeline.compare_even_odd("Const", "GaussianStim2", 0.01)
    # pipeline.compare_even_odd("Const", "GaussianStim3", 0.01)
    # pipeline.compare_even_odd("Const", "GaussianStim4", 0.01)
    # pipeline.compare_even_odd("GaussianStim1", "GaussianStim", 0.01)
    # pipeline.compare_even_odd("GaussianStim2", "GaussianStim", 0.01)
    # pipeline.compare_even_odd("GaussianStim3", "GaussianStim", 0.01)
    # pipeline.compare_even_odd("GaussianStim4", "GaussianStim", 0.01)

    # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    # save_dir = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    # # save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recall_trials/"
    # # path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/"

    # # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    # data_processor = analysis.DataProcessor(
    #     path_to_data, cell_range, window=[0, 1500])
    # solver_params = {
    #     "niter": 10,
    #     "stepsize": 1000,
    #     "interval": 5,
    #     "method": "TNC",
    #     "use_jac": True,
    #     "T" : 1,
    #     "disp":False
    # }
    # bounds_smtstim = {
    #     "sigma": [0, 1000.],
    #     "mu": [0, 1500.],
    #     "tau": [20, 2000.],
    #     "a_1": [0., 1/5.],
    #     "a_2": [0., 1/5.],
    #     "a_3": [0., 1/5.],
    #     "a_4": [0., 1/5.],
    #     "a_0": [0., 1/5.]
    # }
    # bounds_smt = {
    #     "sigma": [0, 1000.],
    #     "mu": [0, 1500.],
    #     "tau": [20, 2000.],
    #     "a_1": [0., 1/2.],
    #     "a_0": [0., 1/2.]
    # }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "Const","SigmaMuTau", "SigmaMuTauStimWarden", "SigmaMuTauStim1", \
    #         "SigmaMuTauStim2", "SigmaMuTauStim3", "SigmaMuTauStim4"], save_dir=save_dir)
    # pipeline.set_model_bounds("SigmaMuTau", bounds_smt)
    # pipeline.set_model_bounds("SigmaMuTauStim1", bounds_smt)
    # pipeline.set_model_bounds("SigmaMuTauStim2", bounds_smt)
    # pipeline.set_model_bounds("SigmaMuTauStim3", bounds_smt)
    # pipeline.set_model_bounds("SigmaMuTauStim4", bounds_smt)
    # pipeline.set_model_bounds("SigmaMuTauStimWarden", bounds_smtstim)
    # pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # with open("/Users/stevecharczynski/workspace/data/warden/recall_trials/info.json") as f:
    # # with open("/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/info.json") as f:
    #     stims = json.load(f)
    #     stims = {int(k):v for k,v in stims.items()}
    # pipeline.set_model_info("SigmaMuTauStimWarden", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("SigmaMuTauStim1", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("SigmaMuTauStim2", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("SigmaMuTauStim3", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("SigmaMuTauStim4", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_x0("SigmaMuTauStimWarden", [10, 1000, 100, 1e-1, 1e-1,1e-1, 1e-1, 1e-1])
    # pipeline.set_model_x0("SigmaMuTau", [10, 1000, 100, 1e-1, 1e-1])
    # pipeline.set_model_x0("SigmaMuTauStim1", [10, 1000, 100, 1e-1, 1e-1])
    # pipeline.set_model_x0("SigmaMuTauStim2", [10, 1000, 100, 1e-1, 1e-1])
    # pipeline.set_model_x0("SigmaMuTauStim3", [10, 1000, 100, 1e-1, 1e-1])
    # pipeline.set_model_x0("SigmaMuTauStim4", [10, 1000, 100, 1e-1, 1e-1])
    # pipeline.set_model_x0("Const", [1e-1])
    # pipeline.fit_all_models(solver_params=solver_params)
    # # pipeline.fit_even_odd(solver_params=solver_params)
    # # pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    # # pipeline.compare_even_odd("SigmaMuTau", "SigmaMuTauStimWarden", 0.01)
    # pipeline.compare_models("Const", "SigmaMuTau", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "SigmaMuTauStimWarden", 0.01, smoother_value=100)
    # pipeline.compare_models("SigmaMuTau", "SigmaMuTauStimWarden", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "SigmaMuTauStim1", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "SigmaMuTauStim2", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "SigmaMuTauStim3", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "SigmaMuTauStim4", 0.01, smoother_value=100)
    # pipeline.compare_models("SigmaMuTauStim1", "SigmaMuTauStimWarden", 0.01, smoother_value=100)
    # pipeline.compare_models("SigmaMuTauStim2", "SigmaMuTauStimWarden", 0.01, smoother_value=100)
    # pipeline.compare_models("SigmaMuTauStim3", "SigmaMuTauStimWarden", 0.01, smoother_value=100)
    # pipeline.compare_models("SigmaMuTauStim4", "SigmaMuTauStimWarden", 0.01, smoother_value=100)
    # pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    # pipeline.compare_even_odd("Const", "SigmaMuTauStimWarden", 0.01)
    # pipeline.compare_even_odd("SigmaMuTau", "SigmaMuTauStimWarden", 0.01)
    # pipeline.compare_even_odd("Const", "SigmaMuTauStim1", 0.01)
    # pipeline.compare_even_odd("Const", "SigmaMuTauStim2", 0.01)
    # pipeline.compare_even_odd("Const", "SigmaMuTauStim3", 0.01)
    # pipeline.compare_even_odd("Const", "SigmaMuTauStim4", 0.01)
    # pipeline.compare_even_odd("SigmaMuTauStim1", "SigmaMuTauStimWarden", 0.01)
    # pipeline.compare_even_odd("SigmaMuTauStim2", "SigmaMuTauStimWarden", 0.01)
    # pipeline.compare_even_odd("SigmaMuTauStim3", "SigmaMuTauStimWarden", 0.01)
    # pipeline.compare_even_odd("SigmaMuTauStim4", "SigmaMuTauStimWarden", 0.01)

    # # pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    # pipeline.compare_even_odd("Const", "SigmaMuTauStimWarden", 0.01)
    # pipeline.compare_even_odd("SigmaMuTau", "SigmaMuTauStimWarden", 0.01)   

    # # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    # # save_dir = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    # save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recall_trials/"
    # path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/"

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
    # # with open("/Users/stevecharczynski/workspace/data/warden/recall_trials/info.json") as f:
    # pipeline.set_model_x0("SigmaMuTau", [0.01, 1000, 100, 1e-1, 1e-1])
    # pipeline.set_model_x0("Const", [1e-1])
    # pipeline.fit_all_models(solver_params=solver_params)
    # pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "SigmaMuTau", 0.01)
    # pipeline.compare_models("Const", "SigmaMuTau", 0.01, smoother_value=100)

# run_script(range(12,13))
if __name__ == "__main__":
    cell_range = range(int(sys.argv[1]), int(sys.argv[2])+1)
    run_script(cell_range)