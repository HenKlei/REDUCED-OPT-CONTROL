import time

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.hapod import dist_vectorarray_hapod, inc_vectorarray_hapod
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from ml_control.completely_reduced_model import CompletelyReducedModel
from ml_control.double_greedy_algorithm import double_greedy
from ml_control.greedy_algorithm import greedy
from ml_control.reduced_model import ReducedModel


def setup_initial_system_bases(full_model, initial_parameter, reduction_strategy, reduction_strategy_parameters):
    phi = full_model.solve(initial_parameter)

    assert reduction_strategy in ["inc_hapod", "dist_hapod"]

    np_vector_space = NumpyVectorSpace(full_model.n)
    u = full_model.compute_control(initial_parameter, phi)
    primal_trajectory = full_model.compute_state(initial_parameter, u)
    np_vector_array_pr = np_vector_space.from_numpy(primal_trajectory)
    if reduction_strategy == "inc_hapod":
        modes_pr, svals_pr, _ = inc_vectorarray_hapod(reduction_strategy_parameters["primal"]["steps"],
                                                      np_vector_array_pr,
                                                      reduction_strategy_parameters["primal"]["eps"],
                                                      reduction_strategy_parameters["primal"]["omega"],
                                                      product=full_model.G_product)
    elif reduction_strategy == "dist_hapod":
        modes_pr, svals_pr, _ = dist_vectorarray_hapod(reduction_strategy_parameters["primal"]["num_slices"],
                                                       np_vector_array_pr,
                                                       reduction_strategy_parameters["primal"]["eps"],
                                                       reduction_strategy_parameters["primal"]["omega"],
                                                       product=full_model.G_product)

    Vpr = modes_pr.to_numpy().T
    assert Vpr.shape[0] == full_model.n
    Wpr = Vpr

    adjoint_trajectory = full_model.compute_adjoint(initial_parameter, phi)
    np_vector_array_ad = np_vector_space.from_numpy(adjoint_trajectory)

    if reduction_strategy == "inc_hapod":
        modes_ad, svals_ad, _ = inc_vectorarray_hapod(reduction_strategy_parameters["adjoint"]["steps"],
                                                      np_vector_array_ad,
                                                      reduction_strategy_parameters["adjoint"]["eps"],
                                                      reduction_strategy_parameters["adjoint"]["omega"],
                                                      product=full_model.G_product)
    elif reduction_strategy == "dist_hapod":
        modes_ad, svals_ad, _ = dist_vectorarray_hapod(reduction_strategy_parameters["adjoint"]["num_slices"],
                                                       np_vector_array_ad,
                                                       reduction_strategy_parameters["adjoint"]["eps"],
                                                       reduction_strategy_parameters["adjoint"]["omega"],
                                                       product=full_model.G_product)

    Vad = modes_ad.to_numpy().T
    assert Vad.shape[0] == full_model.n
    Wad = Vad

    return Vpr, Wpr, Vad, Wad, svals_pr, svals_ad


def _greedy_on_completely_reduced(completely_reduced_model, full_model, training_parameters_greedy, tol,
                                  max_basis_size=None, return_errors_and_efficiencies=False, optimal_adjoints=None):
    results_greedy = greedy(
            training_parameters_greedy, full_model, completely_reduced_model, tol=tol, max_basis_size=max_basis_size,
            return_errors_and_efficiencies=return_errors_and_efficiencies, optimal_adjoints=optimal_adjoints)

    reduced_model = ReducedModel(completely_reduced_model.reduced_basis, full_model)

    results_greedy['tol'] = tol
    results_greedy['training_parameters_greedy'] = training_parameters_greedy
    results_greedy['reduced_basis'] = completely_reduced_model.reduced_basis

    return reduced_model, completely_reduced_model, results_greedy


def greedy_on_completely_reduced_model(full_model, initial_parameter, reduction_strategy, reduction_strategy_parameters,
                                       training_parameters_greedy, tol, max_basis_size=None,
                                       return_errors_and_efficiencies=False, optimal_adjoints=None):
    tic = time.perf_counter()
    Vpr, Wpr, Vad, Wad, svals_pr, svals_ad = setup_initial_system_bases(full_model, initial_parameter,
                                                                        reduction_strategy,
                                                                        reduction_strategy_parameters)
    completely_reduced_model = CompletelyReducedModel(Vpr, Wpr, Vad, Wad, [], full_model,
                                                      reduction_strategy=reduction_strategy,
                                                      reduction_strategy_parameters=reduction_strategy_parameters,
                                                      svals_pr=svals_pr, svals_ad=svals_ad)
    initialization_time = time.perf_counter() - tic

    reduced_model, completely_reduced_model, results_greedy = _greedy_on_completely_reduced(
            completely_reduced_model, full_model, training_parameters_greedy, tol, max_basis_size,
            return_errors_and_efficiencies=return_errors_and_efficiencies, optimal_adjoints=optimal_adjoints)

    offline_timings = {'required_time_overall': results_greedy['required_time_greedy'] + initialization_time,
                       'required_time_greedy': results_greedy['required_time_greedy'],
                       'required_time_system_reductions': None}

    return reduced_model, completely_reduced_model, results_greedy, offline_timings


def greedy_on_reduced_model_and_completely_reduced_construction_afterwards(
        full_model, reduction_strategy, reduction_strategy_parameters, training_parameters_greedy, tol, max_basis_size,
        training_parameters_state_reduction_pr, training_parameters_state_reduction_ad,
        return_errors_and_efficiencies=False, optimal_adjoints=None):
    reduced_model = ReducedModel([], full_model)
    results_greedy = greedy(
            training_parameters_greedy, full_model, reduced_model, tol=tol, max_basis_size=max_basis_size,
            return_errors_and_efficiencies=return_errors_and_efficiencies, optimal_adjoints=optimal_adjoints)

    reduced_model.summary()

    tic = time.perf_counter()
    np_vector_space = NumpyVectorSpace(full_model.n)
    np_vector_array_pr = np_vector_space.empty(reserve=full_model.nt * len(training_parameters_state_reduction_pr))

    for mu in training_parameters_state_reduction_pr:
        phi_mu = full_model.solve(mu)
        u_mu = full_model.compute_control(mu, phi_mu)
        primal_trajectory = full_model.compute_state(mu, u_mu)
        np_vector_array_pr.append(np_vector_space.from_numpy(primal_trajectory))

    assert reduction_strategy in ["inc_hapod", "dist_hapod"]

    if reduction_strategy == "inc_hapod":
        modes_pr, svals_pr, _ = inc_vectorarray_hapod(reduction_strategy_parameters["primal"]["steps"],
                                                      np_vector_array_pr,
                                                      reduction_strategy_parameters["primal"]["eps"],
                                                      reduction_strategy_parameters["primal"]["omega"],
                                                      product=NumpyMatrixOperator(full_model.G))
    elif reduction_strategy == "dist_hapod":
        modes_pr, svals_pr, _ = dist_vectorarray_hapod(reduction_strategy_parameters["primal"]["num_slices"],
                                                       np_vector_array_pr,
                                                       reduction_strategy_parameters["primal"]["eps"],
                                                       reduction_strategy_parameters["primal"]["omega"],
                                                       product=NumpyMatrixOperator(full_model.G))

    Vpr = modes_pr.to_numpy().T
    assert Vpr.shape[0] == full_model.n
    Wpr = Vpr

    np_vector_array_ad = np_vector_space.empty(reserve=full_model.nt * len(training_parameters_state_reduction_ad))

    for mu in training_parameters_state_reduction_ad:
        phi_mu = full_model.solve(mu)
        adjoint_trajectory = full_model.compute_adjoint(mu, phi_mu)
        np_vector_array_ad.append(np_vector_space.from_numpy(adjoint_trajectory))

    if reduction_strategy == "inc_hapod":
        modes_ad, svals_ad, _ = inc_vectorarray_hapod(reduction_strategy_parameters["adjoint"]["steps"],
                                                      np_vector_array_ad,
                                                      reduction_strategy_parameters["adjoint"]["eps"],
                                                      reduction_strategy_parameters["adjoint"]["omega"],
                                                      product=NumpyMatrixOperator(full_model.G))
    elif reduction_strategy == "dist_hapod":
        modes_ad, svals_ad, _ = dist_vectorarray_hapod(reduction_strategy_parameters["adjoint"]["num_slices"],
                                                       np_vector_array_ad,
                                                       reduction_strategy_parameters["adjoint"]["eps"],
                                                       reduction_strategy_parameters["adjoint"]["omega"],
                                                       product=NumpyMatrixOperator(full_model.G))

    current_num_ad = len(modes_ad)
    modes_ad.append(reduced_model.reduced_basis)
    modes_ad = gram_schmidt(modes_ad, product=NumpyMatrixOperator(full_model.G), offset=current_num_ad)
    Vad = modes_ad.to_numpy().T
    assert Vad.shape[0] == full_model.n
    Wad = Vad

    completely_reduced_model = CompletelyReducedModel(Vpr, Wpr, Vad, Wad, reduced_model.reduced_basis, full_model,
                                                      reduction_strategy=reduction_strategy,
                                                      reduction_strategy_parameters=reduction_strategy_parameters)
    required_time_system_reductions = time.perf_counter() - tic

    results_greedy['tol'] = tol
    results_greedy['training_parameters_greedy'] = training_parameters_greedy
    results_greedy['reduced_basis'] = completely_reduced_model.reduced_basis
    offline_timings = {'required_time_overall': required_time_system_reductions+results_greedy['required_time_greedy'],
                       'required_time_greedy': results_greedy['required_time_greedy'],
                       'required_time_system_reductions': required_time_system_reductions}

    return reduced_model, completely_reduced_model, results_greedy, offline_timings


def state_reduction_and_greedy_for_final_time_adjoints_afterwards(
        full_model, training_parameters_state_reduction_pr, training_parameters_state_reduction_ad,
        training_parameters_greedy, reduction_strategy, reduction_strategy_parameters, tol, max_basis_size=None,
        return_errors_and_efficiencies=False, optimal_adjoints=None):
    tic = time.perf_counter()

    np_vector_space = NumpyVectorSpace(full_model.n)
    np_vector_array_ad = np_vector_space.empty(reserve=full_model.nt * len(training_parameters_state_reduction_ad))
    np_vector_array_pr = np_vector_space.empty(reserve=full_model.nt * len(training_parameters_state_reduction_pr))

    for mu in training_parameters_state_reduction_pr:
        phi_mu = full_model.solve(mu)
        adjoint_trajectory = full_model.compute_adjoint(mu, phi_mu)
        np_vector_array_ad.append(np_vector_space.from_numpy(adjoint_trajectory))
        u_mu = full_model.compute_control(mu, phi_mu)
        primal_trajectory = full_model.compute_state(mu, u_mu)
        np_vector_array_pr.append(np_vector_space.from_numpy(primal_trajectory))

    assert reduction_strategy in ["inc_hapod", "dist_hapod"]

    if reduction_strategy == "inc_hapod":
        modes_pr, svals_pr, _ = inc_vectorarray_hapod(reduction_strategy_parameters["primal"]["steps"],
                                                      np_vector_array_pr,
                                                      reduction_strategy_parameters["primal"]["eps"],
                                                      reduction_strategy_parameters["primal"]["omega"],
                                                      product=full_model.G_product)
    elif reduction_strategy == "dist_hapod":
        modes_pr, svals_pr, _ = dist_vectorarray_hapod(reduction_strategy_parameters["primal"]["num_slices"],
                                                       np_vector_array_pr,
                                                       reduction_strategy_parameters["primal"]["eps"],
                                                       reduction_strategy_parameters["primal"]["omega"],
                                                       product=full_model.G_product)

    Vpr = modes_pr.to_numpy().T
    assert Vpr.shape[0] == full_model.n
    Wpr = Vpr

    if reduction_strategy == "inc_hapod":
        modes_ad, svals_ad, _ = inc_vectorarray_hapod(reduction_strategy_parameters["adjoint"]["steps"],
                                                      np_vector_array_ad,
                                                      reduction_strategy_parameters["adjoint"]["eps"],
                                                      reduction_strategy_parameters["adjoint"]["omega"],
                                                      product=full_model.G_product)
    elif reduction_strategy == "dist_hapod":
        modes_ad, svals_ad, _ = dist_vectorarray_hapod(reduction_strategy_parameters["adjoint"]["num_slices"],
                                                       np_vector_array_ad,
                                                       reduction_strategy_parameters["adjoint"]["eps"],
                                                       reduction_strategy_parameters["adjoint"]["omega"],
                                                       product=full_model.G_product)

    Vad = modes_ad.to_numpy().T
    assert Vad.shape[0] == full_model.n
    Wad = Vad

    completely_reduced_model = CompletelyReducedModel(Vpr, Wpr, Vad, Wad, [], full_model,
                                                      reduction_strategy=reduction_strategy,
                                                      reduction_strategy_parameters=reduction_strategy_parameters,
                                                      extend_state_system_basis_on_extend=False, svals_pr=svals_pr, svals_ad=svals_ad)
    required_time_system_reductions = time.perf_counter() - tic

    reduced_model, completely_reduced_model, results_greedy = _greedy_on_completely_reduced(
            completely_reduced_model, full_model, training_parameters_greedy, tol, max_basis_size,
            return_errors_and_efficiencies=return_errors_and_efficiencies, optimal_adjoints=optimal_adjoints)

    offline_timings = {'required_time_overall': required_time_system_reductions+results_greedy['required_time_greedy'],
                       'required_time_greedy': results_greedy['required_time_greedy'],
                       'required_time_system_reductions': required_time_system_reductions}

    return reduced_model, completely_reduced_model, results_greedy, offline_timings


def double_greedy_on_completely_reduced_model(
        full_model, initial_parameter, reduction_strategy, reduction_strategy_parameters, tol, inner_tol,
        training_parameters_greedy, training_parameters_inner_greedy, max_basis_size=None, max_inner_iterations=None,
        return_errors_and_efficiencies=False, optimal_adjoints=None):
    tic = time.perf_counter()
    Vpr, Wpr, Vad, Wad, svals_pr, svals_ad = setup_initial_system_bases(full_model, initial_parameter, reduction_strategy,
                                                    reduction_strategy_parameters)
    completely_reduced_model = CompletelyReducedModel(Vpr, Wpr, Vad, Wad, [], full_model,
                                                      reduction_strategy=reduction_strategy,
                                                      reduction_strategy_parameters=reduction_strategy_parameters,
                                                      svals_pr=svals_pr, svals_ad=svals_ad)
    initialization_time = time.perf_counter() - tic

    results_greedy = double_greedy(training_parameters_greedy, training_parameters_inner_greedy, full_model,
                                   completely_reduced_model, tol=tol, inner_tol=inner_tol,
                                   max_basis_size=max_basis_size, max_inner_iterations=max_inner_iterations,
                                   return_errors_and_efficiencies=return_errors_and_efficiencies,
                                   optimal_adjoints=optimal_adjoints)

    offline_timings = {'required_time_overall': results_greedy['required_time_greedy'] + initialization_time,
                       'required_time_greedy': results_greedy['required_time_greedy'],
                       'required_time_system_reductions': None}

    return completely_reduced_model, results_greedy, offline_timings
