import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting
import json

def run_script(cell_range):
     # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # save_dir = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recall_trials_first_stim_time/"
    path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/"
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

    bounds_stim = {
        "sigma": [1e-4, 1000.],
        "mu": [0, 1500.],
        "a_1": [1e-10, 1/2.],
        "a_0": [1e-10, 1/2.]
    }
    pipeline = analysis.Pipeline(cell_range, data_processor, [
        "Const","GaussianStim1", \
            "GaussianStim2", "GaussianStim3", "GaussianStim4"], save_dir=save_dir)
    pipeline.set_model_bounds("GaussianStim1", bounds_stim)
    pipeline.set_model_bounds("GaussianStim2", bounds_stim)
    pipeline.set_model_bounds("GaussianStim3", bounds_stim)
    pipeline.set_model_bounds("GaussianStim4", bounds_stim)
    pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # with open("/Users/stevecharczynski/workspace/data/warden/recog_trials/info.json") as f:
    with open("/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/info.json") as f:
        stims = json.load(f)
        stims = {int(k):v for k,v in stims.items()}
    pipeline.set_model_info("GaussianStim1", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("GaussianStim2", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("GaussianStim3", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("GaussianStim4", "stim_identity", stims, per_cell=True)
    pipeline.set_model_x0("GaussianStim1", [10, 10, 1e-1, 1e-1])
    pipeline.set_model_x0("GaussianStim2", [10, 10, 1e-1, 1e-1])
    pipeline.set_model_x0("GaussianStim3", [10, 10, 1e-1, 1e-1])
    pipeline.set_model_x0("GaussianStim4", [10, 10, 1e-1, 1e-1])
    pipeline.set_model_x0("Const", [1e-1])
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "Gaussian", 0.01)
    # pipeline.compare_even_odd("Gaussian", "GaussianStimWarden", 0.01)
    pipeline.compare_models("Const", "GaussianStim1", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "GaussianStim2", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "GaussianStim3", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "GaussianStim4", 0.01, smoother_value=100)
    pipeline.compare_even_odd("Const", "GaussianStim1", 0.01)
    pipeline.compare_even_odd("Const", "GaussianStim2", 0.01)
    pipeline.compare_even_odd("Const", "GaussianStim3", 0.01)
    pipeline.compare_even_odd("Const", "GaussianStim4", 0.01)

# run_script(range(12,13))
if __name__ == "__main__":
    cell_range = range(int(sys.argv[1]), int(sys.argv[2])+1)
    run_script(cell_range)