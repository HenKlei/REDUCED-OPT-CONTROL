import numpy as np
import time
import pathlib
from vkoga.kernels import Gaussian

from ml_control.analysis import run_greedy_and_analysis, write_results_to_file
from ml_control.problem_definitions.wave_equation import create_wave_equation_problem
from ml_control.logger import getLogger
from ml_control.machine_learning_models.gaussian_process_regression import GaussianProcessRegressionReducedModel
from ml_control.machine_learning_models.kernel_reduced_model import KernelReducedModel
from ml_control.machine_learning_models.neural_network_reduced_model import NeuralNetworkReducedModel


full_model = create_wave_equation_problem(damping_force=10.)
parameter_space = full_model.parameter_space

k_train = 50
training_parameters = parameter_space.sample_uniformly(k_train)
k_test_plotting = 5
test_parameters_plotting = parameter_space.sample_randomly(k_test_plotting)
test_parameters_plotting = np.sort(test_parameters_plotting, axis=0)
k_test_analysis = 100
test_parameters_analysis = parameter_space.sample_randomly(k_test_analysis)
test_parameters_analysis = np.sort(test_parameters_analysis, axis=0)
tol = 1e-2
max_basis_size = min(30, k_train)
logger = getLogger('damped_wave_equation', level='INFO')

results_greedy, results_analysis = run_greedy_and_analysis(training_parameters, test_parameters_plotting,
                                                           test_parameters_analysis, full_model, tol, max_basis_size,
                                                           logger,
                                                           ml_roms_list=[(NeuralNetworkReducedModel, "DNN-ROM", {}),
                                                                         (KernelReducedModel, "VKOGA-ROM",
                                                                          {"kernel": Gaussian(1.0)}),
                                                                         (GaussianProcessRegressionReducedModel,
                                                                          "GPR-ROM", {})])

write_results = True
if write_results:
    filepath_prefix = 'results_damped/results_wave_equation_' + time.strftime('%Y%m%d-%H%M%S') + '/'
    pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)
    write_results_to_file(results_greedy, results_analysis, filepath_prefix)
