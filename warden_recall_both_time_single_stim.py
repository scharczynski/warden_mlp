import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting
import json

def run_script(cell_range):
    # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    # save_dir = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recall_trials_both_main_time_single_stim/"
    path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/"
    data_processor = analysis.DataProcessor(
        path_to_data, cell_range, window=[0, 3000])
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
        "sigma1": [1e-4, 1000.],
        "mu1": [0, 3000.],
        "a_1": [1e-10, 1/5.],
        "a_2": [1e-10, 1/5.],
        "a_3": [1e-10, 1/5.],
        "a_4": [1e-10, 1/5.],
        "a_0": [1e-10, 1/5.]
    }
    bounds_stim_pos = {
        "sigma1": [1e-4, 1000.],
        "mu1": [0, 3000.],
        "a_1": [1e-10, 1/9.],
        "a_2": [1e-10, 1/9.],
        "a_3": [1e-10, 1/9.],
        "a_4": [1e-10, 1/9.],
        "a_5": [1e-10, 1/9.],
        "a_6": [1e-10, 1/9.],
        "a_7": [1e-10, 1/9.],
        "a_8": [1e-10, 1/9.],
        "a_0": [1e-10, 1/9.]
    }

    bounds_pos = {
        "sigma1": [1e-4, 1000.],
        "mu1": [0, 3000.],
        "a_1": [1e-10, 1/3.],
        "a_2": [1e-10, 1/3.],
        "a_0": [1e-10, 1/3.]
    }

    bounds_time = {
        "sigma1": [1e-4, 1000.],
        "mu1": [0, 3000.],
        "a_1": [1e-10, 1/2.],
        "a_0": [1e-10, 1/2.]
    }

    x0_time =  [10, 100, 1e-1, 1e-1]
    x0_pos = [10, 100, 1e-1, 1e-1, 1e-1]
    x0_stim_pos = [10, 100, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,1e-1, 1e-1,1e-1, 1e-1]
    x0_stim  = [10, 100, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]
    pipeline = analysis.Pipeline(cell_range, data_processor, [
        "Const","GaussianBoth", "GaussianFirstStimBoth", "GaussianSecondStimBoth"], save_dir=save_dir)
    pipeline.set_model_bounds("GaussianBoth", bounds_time)
    pipeline.set_model_bounds("GaussianFirstStimBoth", bounds_stim)
    pipeline.set_model_bounds("GaussianSecondStimBoth", bounds_stim)
    pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # with open("/Users/stevecharczynski/workspace/data/warden/recall_trials/info.json") as f:
    with open("/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/info.json") as f:
        stims = json.load(f)
        stims = {int(k):v for k,v in stims.items()}
    pipeline.set_model_info("GaussianFirstStimBoth", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("GaussianSecondStimBoth", "stim_identity", stims, per_cell=True)
    pipeline.set_model_x0("GaussianFirstStimBoth", x0_stim)
    pipeline.set_model_x0("GaussianSecondStimBoth", x0_stim)
    pipeline.set_model_x0("GaussianBoth", x0_time)
    pipeline.set_model_x0("Const", [1e-1])
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.fit_even_odd(solver_params=solver_params)
    pipeline.compare_models("Const", "GaussianBoth", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "GaussianFirstStimBoth", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "GaussianSecondStimBoth", 0.01, smoother_value=100)

    pipeline.compare_models("GaussianBoth", "GaussianFirstStimBoth", 0.01, smoother_value=100)
    pipeline.compare_models("GaussianBoth", "GaussianSecondStimBoth", 0.01, smoother_value=100)

    pipeline.compare_even_odd("Const", "GaussianBoth", 0.01)
    pipeline.compare_even_odd("Const", "GaussianFirstStimBoth", 0.01)
    pipeline.compare_even_odd("Const", "GaussianSecondStimBoth", 0.01)

    pipeline.compare_even_odd("GaussianBoth", "GaussianFirstStimBoth", 0.01)
    pipeline.compare_even_odd("GaussianBoth", "GaussianSecondStimBoth", 0.01)

   
# run_script((1,1))
if __name__ == "__main__":
    cell_range = range(int(sys.argv[1]), int(sys.argv[2])+1)
    run_script(cell_range)
