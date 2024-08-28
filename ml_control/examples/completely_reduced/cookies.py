from ml_control.analysis import run_reduction_procedures_and_analysis
from ml_control.logger import getLogger
from ml_control.problem_definitions.cookies import create_cookies_problem


logger = getLogger("analysis", level='INFO')

nt = 50
full_model = create_cookies_problem(nt, problem_size="medium")
parameter_space = full_model.parameter_space

tol = 1e-4
inner_tol_double_greedy = 1e-5

reduction_strategy = "dist_hapod"
reduction_strategy_parameters = {"primal": {"num_slices": 50, "eps": 1e-9, "omega": 0.9},
                                 "adjoint": {"num_slices": 50, "eps": 1e-9, "omega": 0.9}}

initial_parameter = [1, 1]

# training parameters system reductions
k_train_state_reduction = 10
training_parameters_state_reduction = parameter_space.sample_uniformly(k_train_state_reduction)

# training parameters greedy
k_train_greedy = 20
training_parameters_greedy = parameter_space.sample_uniformly(k_train_greedy)
max_basis_size = len(training_parameters_greedy)
training_parameters_inner_greedy = training_parameters_greedy
max_inner_iterations = len(training_parameters_inner_greedy)

# test parameters
k_test = 50
test_parameters = parameter_space.sample_randomly(k_test)

run_reduction_procedures_and_analysis(full_model, logger, tol, inner_tol_double_greedy, max_basis_size,
                                      max_inner_iterations, reduction_strategy, reduction_strategy_parameters,
                                      initial_parameter, training_parameters_state_reduction,
                                      training_parameters_greedy, training_parameters_inner_greedy, test_parameters)
