import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting
import json

def run_script(cell_range):
    # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    # save_dir = "/Users/stevecharczynski/workspace/data/warden/recog_trials/"
    save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recog_trials_conjunctive/"
    path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/"
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
        "sigma1": [1e-3, 1000.],
        "mu1": [0, 3000.],
        "a_1": [1e-10, 1/5.],
        "a_2": [1e-10, 1/5.],
        "a_3": [1e-10, 1/5.],
        "a_4": [1e-10, 1/5.],
        "a_0": [1e-10, 1/5.]
    }
    bounds_con = {
        "sigma1": [1e-3, 1000.],
        "mu1": [0, 3000.],
        "a_1": [1e-10, 1/13.],
        "a_2": [1e-10, 1/13.],
        "a_3": [1e-10, 1/13.],
        "a_4": [1e-10, 1/13.],
        "a_5": [1e-10, 1/13.],
        "a_6": [1e-10, 1/13.],
        "a_7": [1e-10, 1/13.],
        "a_8": [1e-10, 1/13.],
        "a_9": [1e-10, 1/13.],
        "a_10": [1e-10, 1/13.],
        "a_11": [1e-10, 1/13.],
        "a_12": [1e-10, 1/13.],
        "a_0": [1e-10, 1/13.]
    }

    bounds_pos = {
        "sigma1": [1e-3, 1000.],
        "mu1": [0, 3000.],
        "a_1": [1e-10, 1/3.],
        "a_2": [1e-10, 1/3.],
        "a_0": [1e-10, 1/3.]
    }

    bounds_time = {
        "sigma1": [1e-3, 1000.],
        "mu1": [0, 2500.],
        "a_1": [1e-10, 1/2.],
        "a_0": [1e-10, 1/2.]
    }

    x0_time =  [10, 100, 1e-1, 1e-1]
    x0_pos = [10, 100, 1e-1, 1e-1, 1e-1]
    x0_con = [10, 100, 1e-1, 1e-1, 1e-1, 1e-1,1e-1, 1e-1, 1e-1, 1e-1, 1e-1,1e-1, 1e-1,1e-1, 1e-1]
    x0_stim  = [10, 100, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1]
    pipeline = analysis.Pipeline(cell_range, data_processor, [
        "Const","Gaussian", "GaussianConjunctive"], save_dir=save_dir)
    pipeline.set_model_bounds("GaussianConjunctive", bounds_con)
    pipeline.set_model_bounds("Gaussian", bounds_time)
    pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # with open("/Users/stevecharczynski/workspace/data/warden/recog_trials/info.json") as f:
    with open("/projectnb/ecog-eeg/stevechar/data/warden/recog_trials/info.json") as f:
        stims = json.load(f)
        stims = {int(k):v for k,v in stims.items()}
    pipeline.set_model_info("GaussianConjunctive", "stim_identity", stims, per_cell=True)
    pipeline.set_model_x0("GaussianConjunctive", x0_con)
    pipeline.set_model_x0("Gaussian", x0_time)
    pipeline.set_model_x0("Const", [1e-1])
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.fit_even_odd(solver_params=solver_params)
    pipeline.compare_models("Const", "Gaussian", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "GaussianConjunctive", 0.01, smoother_value=100)
    pipeline.compare_models("Gaussian", "GaussianConjunctive", 0.01, smoother_value=100)
    pipeline.compare_even_odd("Const", "GaussianConjunctive", 0.01)
    pipeline.compare_even_odd("Gaussian", "GaussianConjunctive", 0.01)
    pipeline.compare_even_odd("Const", "Gaussian", 0.01)
   
# run_script((1,1))
if __name__ == "__main__":
    cell_range = range(int(sys.argv[1]), int(sys.argv[2])+1)
    run_script(cell_range)
