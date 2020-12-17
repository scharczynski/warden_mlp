import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting
import json

def run_script(cell_range):
    # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # save_dir = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recog_trials_first_stim/"
    path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/"

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
    bounds_smt = {
        "sigma": [1e-4, 3e-4],
        "mu": [0, 1500.],
        "tau": [1e-1, 10000.],
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
    with open("/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/info.json") as f:
        stims = json.load(f)
        stims = {int(k):v for k,v in stims.items()}
    pipeline.set_model_info("SigmaMuTauStim1", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("SigmaMuTauStim2", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("SigmaMuTauStim3", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("SigmaMuTauStim4", "stim_identity", stims, per_cell=True)
    pipeline.set_model_x0("SigmaMuTauStim1", [2e-4, 10, 100, 1e-1, 1e-1])
    pipeline.set_model_x0("SigmaMuTauStim2", [2e-4, 10, 100, 1e-1, 1e-1])
    pipeline.set_model_x0("SigmaMuTauStim3", [2e-4, 10, 100, 1e-1, 1e-1])
    pipeline.set_model_x0("SigmaMuTauStim4", [2e-4, 10, 100, 1e-1, 1e-1])
    pipeline.set_model_x0("Const", [1e-1])
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.fit_even_odd(solver_params=solver_params)
    pipeline.compare_models("Const", "SigmaMuTauStim1", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "SigmaMuTauStim2", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "SigmaMuTauStim3", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "SigmaMuTauStim4", 0.01, smoother_value=100)



# run_script(range(32, 33))
if __name__ == "__main__":
    cell_range = range(int(sys.argv[1]), int(sys.argv[2])+1)
    run_script(cell_range)