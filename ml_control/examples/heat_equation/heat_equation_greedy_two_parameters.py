import time
import pathlib

from ml_control.analysis import run_greedy_and_analysis, write_results_to_file
from ml_control.logger import getLogger
from ml_control.problem_definitions.heat_equation import create_heat_equation_problem_with_two_parameters


full_model = create_heat_equation_problem_with_two_parameters()
parameter_space = full_model.parameter_space

k_train = 8
training_parameters = parameter_space.sample_uniformly(k_train)
k_test_plotting = 2
test_parameters_plotting = parameter_space.sample_uniformly(k_test_plotting)
k_test_analysis = 10
test_parameters_analysis = parameter_space.sample_uniformly(k_test_analysis)
tol = 1e-5
max_basis_size = k_train ** 2
logger = getLogger('heat_equation', level='INFO')

results_greedy, results_analysis = run_greedy_and_analysis(training_parameters, test_parameters_plotting,
                                                           test_parameters_analysis, full_model, tol, max_basis_size,
                                                           logger)

write_results = True
if write_results:
    filepath_prefix = 'results_two_parameters/results_heat_equation_' + time.strftime('%Y%m%d-%H%M%S') + '/'
    pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)
    write_results_to_file(results_greedy, results_analysis, filepath_prefix)
