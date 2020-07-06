import sys
import maxlikespy.analysis as analysis
import os
import maxlikespy.util as util
import maxlikespy.plotting as plotting
import json

def run_script(cell_range):

    # path_to_data = "/Users/stevecharczynski/workspace/data/warden/recall_trials/"
    # save_dir = "/Users/stevecharczynski/workspace/data/warden/recall_trials_both/"
    save_dir = "/projectnb/ecog-eeg/stevechar/ml_runs/warden/recall_trials_both/"
    path_to_data = "/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/"

    # time_info = list(zip(np.zeros(len(trial_length), dtype=int), trial_length))
    data_processor = analysis.DataProcessor(
        path_to_data, cell_range, window=[0, 3000])
    solver_params = {
        "niter": 1000,
        "stepsize": 100,
        "interval": 5,
        "method": "TNC",
        "use_jac": True,
        "T" : 1,
        "disp":False
    }
    bounds_gaussianstim = {
        "sigma1": [1e-5, 1000.],
        "mu1": [0, 1700.],
        "sigma2": [1e-5, 1000.],
        "mu2": [1500, 3200.],
        "a_1": [1e-10, 1/5.],
        "a_2": [1e-10, 1/5.],
        "a_3": [1e-10, 1/5.],
        "a_4": [1e-10, 1/5.],
        "a_0": [1e-10, 1/5.]
    }

    bounds_gaussianstim_pos = {
        "sigma1": [1e-5, 1000.],
        "mu1": [0, 1700.],
        "sigma2": [1e-5, 1000.],
        "mu2": [1500, 3200.],
        "a_1": [1e-10, 1/5.],
        "a_2": [1e-10, 1/5.],
        "a_3": [1e-10, 1/5.],
        "a_4": [1e-10, 1/5.],
        "a_5": [1e-10, 1/5.],
        "a_6": [1e-10, 1/5.],
        "a_7": [1e-10, 1/5.],
        "a_8": [1e-10, 1/5.],
        "a_0": [1e-10, 1/5.]
    }

    bounds_gaussian = {
        "sigma1": [1e-5, 1000.],
        "mu1": [0, 1700.],
        "sigma2": [1e-5, 1000.],
        "mu2": [1500, 3200.],
        "a_1": [1e-10, 1/2.],
        "a_0": [1e-10, 1/2.]
    }

    bounds_gaussian_pos = {
        "sigma1": [1e-5, 1000.],
        "mu1": [0, 1700.],
        "sigma2": [1e-5, 1000.],
        "mu2": [1500, 3200.],
        "a_1": [1e-10, 1/2.],
        "a_2": [1e-10, 1/2.],
        "a_0": [1e-10, 1/2.]
    }
    # pipeline = analysis.Pipeline(cell_range, data_processor, [
    #     "Const","GaussianBoth", "GaussianStimBoth", "GaussianStimBoth1", \
    #         "GaussianStimBoth2", "GaussianStimBoth3", "GaussianStimBoth4"], save_dir=save_dir)
    pipeline = analysis.Pipeline(cell_range, data_processor, [
        "Const","GaussianBoth", "GaussianStimBoth", "GaussianBothPos", "GaussianStimBothPos"], save_dir=save_dir)
    pipeline.set_model_bounds("GaussianBoth", bounds_gaussian)
    # pipeline.set_model_bounds("GaussianStimBoth1", bounds_gaussian)
    # pipeline.set_model_bounds("GaussianStimBoth2", bounds_gaussian)
    # pipeline.set_model_bounds("GaussianStimBoth3", bounds_gaussian)
    # pipeline.set_model_bounds("GaussianStimBoth4", bounds_gaussian)
    pipeline.set_model_bounds("GaussianBothPos", bounds_gaussian_pos)
    pipeline.set_model_bounds("GaussianStimBothPos", bounds_gaussianstim_pos)
    pipeline.set_model_bounds("GaussianStimBoth", bounds_gaussianstim)
    pipeline.set_model_bounds("Const", {"a_0":[10e-10, 1]})
    # with open("/Users/stevecharczynski/workspace/data/warden/recall_trials/info.json") as f:
    with open("/projectnb/ecog-eeg/stevechar/data/warden/recall_trials/info.json") as f:
        stims = json.load(f)
        stims = {int(k):v for k,v in stims.items()}
    pipeline.set_model_info("GaussianStimBoth", "stim_identity", stims, per_cell=True)
    pipeline.set_model_info("GaussianStimBothPos", "stim_identity", stims, per_cell=True)

    # pipeline.set_model_info("GaussianStimBoth1", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("GaussianStimBoth2", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("GaussianStimBoth3", "stim_identity", stims, per_cell=True)
    # pipeline.set_model_info("GaussianStimBoth4", "stim_identity", stims, per_cell=True)
    pipeline.set_model_x0("GaussianStimBoth", [10, 1000, 10, 2000, 1e-1, 1e-1,1e-1, 1e-1, 1e-1])
    pipeline.set_model_x0("GaussianStimBothPos", [10, 1000, 10, 2000, 1e-1, 1e-1,1e-1, 1e-1, 1e-1, 1e-1,1e-1, 1e-1, 1e-1])
    pipeline.set_model_x0("GaussianBothPos", [10, 1000, 10, 2000, 1e-1, 1e-1,1e-1])

    pipeline.set_model_x0("GaussianBoth", [10, 1000, 10, 2000, 1e-1, 1e-1])
    # pipeline.set_model_x0("GaussianStimBoth1", [10, 1000,10, 2000, 1e-1, 1e-1])
    # pipeline.set_model_x0("GaussianStimBoth2", [10, 1000,10, 2000, 1e-1, 1e-1])
    # pipeline.set_model_x0("GaussianStimBoth3", [10, 1000,10, 2000, 1e-1, 1e-1])
    # pipeline.set_model_x0("GaussianStimBoth4", [10, 1000,10, 2000, 1e-1, 1e-1])
    pipeline.set_model_x0("Const", [1e-1])
    pipeline.fit_all_models(solver_params=solver_params)
    pipeline.fit_even_odd(solver_params=solver_params)
    # pipeline.compare_even_odd("Const", "GaussianBoth", 0.01)
    # pipeline.compare_even_odd("GaussianBoth", "GaussianStimBoth", 0.01)
    pipeline.compare_models("Const", "GaussianBoth", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "GaussianStimBoth", 0.01, smoother_value=100)
    pipeline.compare_models("GaussianBoth", "GaussianStimBoth", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "GaussianBothPos", 0.01, smoother_value=100)
    pipeline.compare_models("Const", "GaussianStimBothPos", 0.01, smoother_value=100)
    pipeline.compare_models("GaussianBoth", "GaussianBothPos", 0.01, smoother_value=100)
    pipeline.compare_models("GaussianBoth", "GaussianStimBothPos", 0.01, smoother_value=100)
    pipeline.compare_models("GaussianBothPos", "GaussianStimBoth", 0.01, smoother_value=100)
    pipeline.compare_models("GaussianBothPos", "GaussianStimBothPos", 0.01, smoother_value=100)
    pipeline.compare_models("GaussianStimBoth", "GaussianStimBothPos", 0.01, smoother_value=100)


    # pipeline.compare_models("Const", "GaussianStimBoth1", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "GaussianStimBoth2", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "GaussianStimBoth3", 0.01, smoother_value=100)
    # pipeline.compare_models("Const", "GaussianStimBoth4", 0.01, smoother_value=100)
    # pipeline.compare_models("GaussianStimBoth1", "GaussianStimBoth", 0.01, smoother_value=100)
    # pipeline.compare_models("GaussianStimBoth2", "GaussianStimBoth", 0.01, smoother_value=100)
    # pipeline.compare_models("GaussianStimBoth3", "GaussianStimBoth", 0.01, smoother_value=100)
    # pipeline.compare_models("GaussianStimBoth4", "GaussianStimBoth", 0.01, smoother_value=100)
    pipeline.compare_even_odd("Const", "GaussianBoth", 0.01)
    pipeline.compare_even_odd("Const", "GaussianStimBoth", 0.01)
    pipeline.compare_even_odd("GaussianBoth", "GaussianStimBoth", 0.01)
    pipeline.compare_even_odd("Const", "GaussianBothPos", 0.01)
    pipeline.compare_even_odd("Const", "GaussianStimBothPos", 0.01)
    pipeline.compare_even_odd("GaussianBoth", "GaussianBothPos", 0.01)
    pipeline.compare_even_odd("GaussianBoth", "GaussianStimBothPos", 0.01)
    pipeline.compare_even_odd("GaussianBothPos", "GaussianStimBoth", 0.01)
    pipeline.compare_even_odd("GaussianBothPos", "GaussianStimBothPos", 0.01)
    pipeline.compare_even_odd("GaussianStimBoth", "GaussianStimBothPos", 0.01)

    # pipeline.compare_even_odd("Const", "GaussianStimBoth1", 0.01)
    # pipeline.compare_even_odd("Const", "GaussianStimBoth2", 0.01)
    # pipeline.compare_even_odd("Const", "GaussianStimBoth3", 0.01)
    # pipeline.compare_even_odd("Const", "GaussianStimBoth4", 0.01)
    # pipeline.compare_even_odd("GaussianStimBoth1", "GaussianStimBoth", 0.01)
    # pipeline.compare_even_odd("GaussianStimBoth2", "GaussianStimBoth", 0.01)
    # pipeline.compare_even_odd("GaussianStimBoth3", "GaussianStimBoth", 0.01)
    # pipeline.compare_even_odd("GaussianStimBoth4", "GaussianStimBoth", 0.01)

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

# run_script(range(15,16))
if __name__ == "__main__":
    cell_range = sys.argv[-2:]
    cell_range = list(map(int, cell_range))
    cell_range = range(cell_range[0], cell_range[1]+1)
    run_script(cell_range)
