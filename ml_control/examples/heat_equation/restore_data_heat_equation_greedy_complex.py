import numpy as np
import pickle
import time
import pathlib
from vkoga.kernels import Gaussian

from ml_control.analysis import run_greedy_and_analysis, write_results_to_file
from ml_control.logger import getLogger
from ml_control.problem_definitions.heat_equation import create_heat_equation_problem_complex
from ml_control.machine_learning_models.gaussian_process_regression import GaussianProcessRegressionReducedModel
from ml_control.machine_learning_models.kernel_reduced_model import KernelReducedModel
from ml_control.machine_learning_models.neural_network_reduced_model import NeuralNetworkReducedModel


full_model = create_heat_equation_problem_complex()
parameter_space = full_model.parameter_space

filepath_prefix = 'results_complex/results_heat_equation_20230619-103119/'

with open(filepath_prefix + 'reduced_basis.pickle', 'rb') as f:
    reduced_basis = pickle.load(f)

with open(filepath_prefix + 'training_data.pickle', 'rb') as f:
    training_data = pickle.load(f)

k_train = 8
training_parameters = np.array(np.meshgrid(np.linspace(*parameter_space[0], k_train),
                                           np.linspace(*parameter_space[1], k_train))).T.reshape(-1, 2)
k_test_plotting = 2
test_parameters_plotting = np.array(np.meshgrid(np.random.uniform(*parameter_space[0], k_test_plotting),
                                                np.random.uniform(*parameter_space[1], k_test_plotting))).T.reshape(-1, 2)
k_test_analysis = 10
test_parameters_analysis = np.array(np.meshgrid(np.random.uniform(*parameter_space[0], k_test_analysis),
                                                np.random.uniform(*parameter_space[1], k_test_analysis))).T.reshape(-1, 2)

cg_params = {}
logger = getLogger('heat_equation', level='INFO')

results_greedy, results_analysis = run_greedy_and_analysis(training_parameters, test_parameters_plotting,
                                                           test_parameters_analysis, full_model,
                                                           None, None, logger,
                                                           ml_roms_list=[(NeuralNetworkReducedModel, "DNN-ROM", {}),
                                                                         (KernelReducedModel, "VKOGA-ROM",
                                                                          {"kernel": Gaussian(1.0)}),
                                                                         (GaussianProcessRegressionReducedModel,
                                                                          "GPR-ROM", {})],
                                                           training_data=training_data, reduced_basis=reduced_basis)

write_results = True
if write_results:
    filepath_prefix = 'results_complex/results_heat_equation_' + time.strftime('%Y%m%d-%H%M%S') + '/'
    pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)
    write_results_to_file(results_greedy, results_analysis, filepath_prefix)
