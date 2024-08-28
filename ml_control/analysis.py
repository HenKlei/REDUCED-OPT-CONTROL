import pathlib
import pickle
import time

from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from ml_control.machine_learning_models.gaussian_process_regression import GaussianProcessRegressionReducedModel
from ml_control.greedy_algorithm import greedy
from ml_control.machine_learning_models.kernel_reduced_model import KernelReducedModel
from ml_control.machine_learning_models.neural_network_reduced_model import NeuralNetworkReducedModel
from ml_control.reduced_model import ReducedModel
from ml_control.visualization import plot_greedy_results, plot_final_time_adjoints, plot_controls, \
        plot_final_time_solutions
from ml_control.reduction_procedures import (greedy_on_completely_reduced_model,
                                             greedy_on_reduced_model_and_completely_reduced_construction_afterwards,
                                             state_reduction_and_greedy_for_final_time_adjoints_afterwards,
                                             double_greedy_on_completely_reduced_model)


def run_greedy_and_analysis(training_parameters, test_parameters_plotting,
                            test_parameters_analysis, full_model, tol, max_basis_size, logger,
                            ml_roms_list=[(NeuralNetworkReducedModel, "DNN-ROM", {}),
                                          (KernelReducedModel, "VKOGA-ROM", {}),
                                          (GaussianProcessRegressionReducedModel, "GPR-ROM", {})],
                            training_data=None, reduced_basis=[]):
    """Runs the greedy algorithm, trains the machine learning surrogates and performs an analysis on a test set."""
    # Offline phase
    restarted_analysis = True
    if training_data is None or not reduced_basis:
        restarted_analysis = False
        rom = ReducedModel(reduced_basis, full_model)
        selected_indices, estimated_errors, true_errors, efficiencies, \
            optimal_adjoints, training_data, required_time = greedy(training_parameters, full_model, rom, tol=tol,
                                                                    max_basis_size=max_basis_size,
                                                                    return_errors_and_efficiencies=True)
        reduced_basis = rom.reduced_basis
        print(f"Reduced basis size: {len(reduced_basis)}")
    else:
        rom = ReducedModel(reduced_basis, full_model)

    ml_roms = []
    labels = []
    for ModelClass, name, training_args in ml_roms_list:
        ml_rom = ModelClass(rom, training_data)
        ml_rom.train(**training_args)
        ml_roms.append(ml_rom)
        labels.append(name)

    # Plotting some results
    if not restarted_analysis:
        logger.info("Computing coefficients ...")
        projection_coefficients = []
        roms_coefficients = [[] for _ in range(1 + len(ml_roms_list))]

        from pymor.vectorarrays.numpy import NumpyVectorSpace
        np_vector_space = NumpyVectorSpace(full_model.n)

        for i, mu in enumerate(training_parameters):
            phiT_mu = full_model.solve(mu)

            alpha_mu = reduced_basis.inner(np_vector_space.from_numpy(phiT_mu), product=full_model.G_product)
            projection_coefficients.append(alpha_mu)

            for j, r in enumerate([rom, *ml_roms]):
                roms_coefficients[j].append(r.solve(mu))

        projection_coefficients = np.array(projection_coefficients)
        for j, r in enumerate(roms_coefficients):
            roms_coefficients[j] = np.array(r)

        _, singular_values, _ = np.linalg.svd(optimal_adjoints)

        plot_greedy_results(training_parameters, selected_indices, estimated_errors, true_errors, efficiencies, tol,
                            [projection_coefficients, roms_coefficients[0], *roms_coefficients[1:]],
                            ["Projection of optimal control", "RB-ROM", *labels],
                            reduced_basis, singular_values)

    # Online phase (plotting)
    with logger.block('Running online phase with plotting of results for '
                      f'{len(test_parameters_plotting)} parameters ...'):
        for i, mu in enumerate(test_parameters_plotting):
            logger.info(f'Results for parameter {mu}:')

            tic = time.perf_counter()
            phi_opt = full_model.solve(mu)
            u_opt = full_model.compute_control(mu, phi_opt)
            time_full = time.perf_counter() - tic
            x_opt = full_model.compute_state(mu, u_opt)
            xT = full_model.parametrized_xT(mu)
            print(f"Deviation from target state for full model: {full_model.spatial_norm(x_opt[-1] - xT)}")

            us_roms = []
            phis_roms = []
            phis_roms_reconstructed = []
            times_roms = []
            xs_final_roms = []

            print(ml_roms)

            for r, name in zip([rom, *ml_roms], ["RB-ROM", *labels]):
                tic = time.perf_counter()
                phi_red = r.solve(mu)
                print(name)
                u_red = r.compute_control(mu, phi_red)
                time_red = time.perf_counter() - tic

                us_roms.append(u_red)
                phis_roms.append(phi_red)
                phis_roms_reconstructed.append(r.reconstruct(phi_red))
                times_roms.append(time_red)

                x = r.compute_state(mu, u_red)
                xs_final_roms.append(x[-1])
                print(f"Deviation from target state ({name}): {full_model.spatial_norm(x[-1] - xT)}")

            for r, phi_red, name in zip([rom, *ml_roms], phis_roms, ["RB-ROM", *labels]):
                print(f"Error in final time adjoint ({name}): "
                      f"{full_model.spatial_norm(r.reconstruct(phi_red) - phi_opt)}")
                print(f"Estimated error in final time adjoint ({name}): {r.estimate_error(mu, phi_red)}")

            for u, name in zip(us_roms, ["RB-ROM", *labels]):
                print(f"Error in control ({name}): {full_model.temporal_norm(u - u_opt)}")

            print(f"Runtime full model: {time_full}")
            for t, name in zip(times_roms, ["RB-ROM", *labels]):
                print(f"Runtime ({name}): {t}")

            for t, name in zip(times_roms, ["RB-ROM", *labels]):
                print(f"Speedup ({name}): {time_full / t}")

            print()

            fig, axs = plt.subplots(3)
            plot_final_time_adjoints([phi_opt, *phis_roms_reconstructed],
                                     labels=["Optimal adjoint", "Reduced adjoint",
                                             *[f"{name} reduced adjoint" for name in labels]],
                                     show_plot=False, ax=axs[0])
            axs[0].set_title("Final time adjoints")
            axs[0].legend()
            plot_controls([u_opt, *us_roms], full_model.T,
                          labels=["Optimal control", "Reduced control",
                                  *[f"{name} reduced control" for name in labels]],
                          show_plot=False, ax=axs[1])
            axs[1].set_title("Controls")
            axs[1].legend()
            plot_final_time_solutions([xT, x_opt[-1], *xs_final_roms],
                                      labels=["Target state", "Optimal state", "Reduced state",
                                              *[f"{name} reduced state" for name in labels]],
                                      show_plot=False, ax=axs[2])
            axs[2].set_title("Final time states")
            axs[2].legend()
            fig.suptitle(f"Results for parameter {mu}")
            plt.show()

    # Online phase (analysis)
    deviations_from_target_state_opt = []
    deviations_from_target_state_roms = [[] for _ in range(1 + len(ml_roms_list))]
    errors_in_final_time_adjoint_roms = [[] for _ in range(1 + len(ml_roms_list))]
    estimated_errors_in_final_time_adjoint_roms = [[] for _ in range(1 + len(ml_roms_list))]
    errors_in_control_roms = [[] for _ in range(1 + len(ml_roms_list))]
    runtimes_opt = []
    runtimes_roms = [[] for _ in range(1 + len(ml_roms_list))]
    speedups_roms = [[] for _ in range(1 + len(ml_roms_list))]

    with logger.block('Running online phase for analysis of results for '
                      f'{len(test_parameters_analysis)} parameters ...'):
        for i, mu in enumerate(test_parameters_analysis):
            logger.info(f'Parameter number {i} ...')

            tic = time.perf_counter()
            phi_opt = full_model.solve(mu)
            u_opt = full_model.compute_control(mu, phi_opt)
            time_full = time.perf_counter() - tic
            x_opt = full_model.compute_state(mu, u_opt)
            deviations_from_target_state_opt.append(full_model.spatial_norm(x_opt[-1] - xT))
            runtimes_opt.append(time_full)

            for i, (r, name) in enumerate(zip([rom, *ml_roms], ["RB-ROM", *labels])):
                tic = time.perf_counter()
                phi_red = r.solve(mu)
                u_red = r.compute_control(mu, phi_red)
                time_red = time.perf_counter() - tic

                x = r.compute_state(mu, u)
                xT = full_model.parametrized_xT(mu)

                deviations_from_target_state_roms[i].append(full_model.spatial_norm(x[-1] - xT))
                errors_in_final_time_adjoint_roms[i].append(full_model.spatial_norm(r.reconstruct(phi_red) - phi_opt))
                estimated_errors_in_final_time_adjoint_roms[i].append(r.estimate_error(mu, phi_red))
                errors_in_control_roms[i].append(full_model.temporal_norm(u_red - u_opt))
                runtimes_roms[i].append(time_red)
                speedups_roms[i].append(time_full / time_red)

    with logger.block(f'================== RESULTS FOR {len(test_parameters_analysis)} PARAMETERS =================='):
        with logger.block('Average deviation from target state:'):
            logger.info(f'Full model: {np.average(deviations_from_target_state_opt)}')
            for i, name in enumerate(["RB-ROM", *labels]):
                logger.info(f'{name}: {np.average(deviations_from_target_state_roms[i])}')
        with logger.block('Average errors in final time adjoint:'):
            for i, name in enumerate(["RB-ROM", *labels]):
                logger.info(f'{name}: {np.average(errors_in_final_time_adjoint_roms[i])}')
        with logger.block('Average errors in control:'):
            for i, name in enumerate(["RB-ROM", *labels]):
                logger.info(f'{name}: {np.average(errors_in_control_roms[i])}')
        with logger.block('Average run time:'):
            logger.info(f'Full model: {np.average(runtimes_opt)}')
            for i, name in enumerate(["RB-ROM", *labels]):
                logger.info(f'{name}: {np.average(runtimes_roms[i])}')
        with logger.block('Average speedup:'):
            for i, name in enumerate(["RB-ROM", *labels]):
                logger.info(f'{name}: {np.average(speedups_roms[i])}')

    fig = plt.figure()
    fig.suptitle('Errors and estimated errors')
    ax = fig.add_subplot(111)
    colors = ["r", "b", "g", "y"]
    for i, (name, c) in enumerate(zip(["RB-ROM", *labels], colors)):
        ax.semilogy(np.arange(len(test_parameters_analysis)), errors_in_final_time_adjoint_roms[i], f"-{c}",
                    label=f"Errors {name}")
        ax.semilogy(np.arange(len(test_parameters_analysis)), estimated_errors_in_final_time_adjoint_roms[i], f"--{c}",
                    label=f"Estimated errors {name}")
    ax.set_xlabel('Test parameter number')
    ax.set_ylabel('Error')
    ax.legend()
    plt.show()

    fig = plt.figure()
    fig.suptitle('Boxplots of errors')
    ax = fig.add_subplot(111)
    ax.boxplot([*errors_in_final_time_adjoint_roms])
    ax.set_yscale('log')
    ax.set_xticklabels(["RB-ROM", *labels])
    ax.set_xlabel('Reduced model')
    ax.set_ylabel('Error')
    plt.show()

    if not restarted_analysis:
        results_greedy = {'tol': tol,
                          'training_parameters': training_parameters,
                          'selected_indices': selected_indices,
                          'estimated_errors': estimated_errors,
                          'true_errors': true_errors,
                          'efficiencies': efficiencies,
                          'singular_values': singular_values,
                          'projection_coefficients': projection_coefficients,
                          'roms_coefficients': roms_coefficients,
                          'reduced_basis': reduced_basis,
                          'training_data': training_data}
    else:
        results_greedy = {'tol': tol,
                          'training_parameters': training_parameters,
                          'reduced_basis': reduced_basis,
                          'training_data': training_data}
    results_analysis = {'test_parameters': test_parameters_analysis,
                        'deviations_from_target_state_opt': deviations_from_target_state_opt,
                        'deviations_from_target_state_roms': deviations_from_target_state_roms,
                        'errors_in_final_time_adjoint_roms': errors_in_final_time_adjoint_roms,
                        'estimated_errors_in_final_time_adjoint_roms': estimated_errors_in_final_time_adjoint_roms,
                        'errors_in_control_roms': errors_in_control_roms,
                        'runtimes_opt': runtimes_opt,
                        'runtimes_roms': runtimes_roms,
                        'speedups_roms': speedups_roms}
    return results_greedy, results_analysis


def write_results_to_file(results_greedy, results_analysis, filepath_prefix,
                          labels=["RB-ROM", "DNN-ROM", "VKOGA-ROM", "GPR-ROM"]):
    """Writes results of greedy algorithm and online analysis to disc."""
    with open(filepath_prefix + 'reduced_basis.pickle', 'wb') as f:
        pickle.dump(results_greedy['reduced_basis'], f)

    with open(filepath_prefix + 'training_data.pickle', 'wb') as f:
        pickle.dump(results_greedy['training_data'], f)

    if 'selected_indices' in results_greedy:
        with open(filepath_prefix + 'results_greedy.txt', 'w') as f:
            f.write("Greedy step\tEstimated errors\tTrue errors\tEfficiencies\tSelected training parameters\n")
            if results_greedy['training_parameters'].ndim == 1:
                selected_params = np.hstack([np.array([None]),
                                             results_greedy['training_parameters'][results_greedy['selected_indices']]])
            else:
                selected_params = np.vstack([np.array([None] * results_greedy['training_parameters'].shape[1]),
                                             results_greedy['training_parameters'][results_greedy['selected_indices']]])
            for i, (e1, e2, e3, e4) in enumerate(zip(results_greedy['estimated_errors'], results_greedy['true_errors'],
                                                     results_greedy['efficiencies'], selected_params)):
                if i == 0:
                    f.write(f"{i}\t{e1}\t{e2}\t{e3}\t ")
                else:
                    f.write(f"{i}\t{e1}\t{e2}\t{e3}\t{e4}")
                f.write("\n")
    if 'singular_values' in results_greedy:
        with open(filepath_prefix + 'singular_values_optimal_adjoints.txt', 'w') as f:
            for i, s in enumerate(results_greedy['singular_values']):
                f.write(f"{i+1}\t{s}\n")

    with open(filepath_prefix + 'analysis_results_summary.txt', 'w') as f:
        for errs, name in zip(results_analysis['errors_in_final_time_adjoint_roms'], labels):
            f.write(f"Maximum error in adjoint ({name}):\t{np.max(errs)}\n")
        f.write("\n")
        for errs, name in zip(results_analysis['errors_in_final_time_adjoint_roms'], labels):
            f.write(f"Average error in adjoint ({name}):\t{np.average(errs)}\n")
        f.write("\n")
        for errs, name in zip(results_analysis['errors_in_control_roms'], labels):
            f.write(f"Maximum error in control ({name}):\t{np.max(errs)}\n")
        f.write("\n")
        for errs, name in zip(results_analysis['errors_in_control_roms'], labels):
            f.write(f"Average error in control ({name}):\t{np.average(errs)}\n")
        f.write("\n")
        f.write(f"Average runtime (Exact solution):\t{np.average(results_analysis['runtimes_opt'])}\n")
        for ts, name in zip(results_analysis['runtimes_roms'], labels):
            f.write(f"Average runtime ({name}):\t{np.average(ts)}\n")
        f.write("\n")
        for s, name in zip(results_analysis['speedups_roms'], labels):
            f.write(f"Average speedup ({name}):\t{np.average(s)}\n")

    with open(filepath_prefix + 'analysis_results_errors.txt', 'w') as f:
        labs = [None] * (2 * len(labels))
        labs[::2] = results_analysis['errors_in_final_time_adjoint_roms']
        labs[1::2] = results_analysis['estimated_errors_in_final_time_adjoint_roms']
        labs.insert(0, results_analysis['test_parameters'])
        for i, (param, e1, e2, e3, e4, e5, e6, e7, e8) in enumerate(zip(*labs)):
            f.write(f"{i+1}\t{param}\t{e1}\t{e2}\t{e3}\t{e4}\t{e5}\t{e6}\t{e7}\t{e8}")
            f.write("\n")


def run_online_tests(model, test_parameters, full_solutions, full_solution_times, full_controls,
                     full_final_time_states, full_model, write_to_file, write_to_test_errors_file):
    solutions = []
    solution_times = []
    controls = []
    errors = []
    errors_control = []
    deviations_from_target = []
    deviations_from_target_using_fom = []
    errors_final_time_states = []
    final_time_states = []
    energies_control = []
    estimated_errors = []
    estimated_errors_components = []
    estimation_times = []
    for i, mu in enumerate(test_parameters):
        tic = time.perf_counter()
        sol = model.solve(mu)
        required_time = time.perf_counter() - tic
        solutions.append(sol)
        solution_times.append(required_time)
        cont = model.compute_control(mu, sol)
        controls.append(cont)
        final_time_state = model.reconstruct_state([model.compute_state(mu, cont)[-1]])[0]
        final_time_states.append(final_time_state)
        final_time_state_of_fom = full_model.compute_state(mu, cont)[-1]
        deviations_from_target.append(full_model.deviation_from_target(final_time_state, mu))
        deviations_from_target_using_fom.append(full_model.deviation_from_target(final_time_state_of_fom, mu))
        energies_control.append(full_model.control_energy(cont))
        if full_solutions:
            errors.append(full_model.compute_weighted_norm(model.reconstruct(sol) - full_solutions[i]))
            errors_control.append(full_model.temporal_norm(cont - full_controls[i]))
            errors_final_time_states.append(full_model.compute_weighted_norm(final_time_state
                                                                             - full_final_time_states[i]))
            tic = time.perf_counter()
            est_err, est_err_comps = model.estimate_error(mu, sol, return_error_components=True)
            required_time = time.perf_counter() - tic
            estimated_errors.append(est_err)
            estimated_errors_components.append(est_err_comps)
            estimation_times.append(required_time)

    average_runtime = np.mean(solution_times)
    write_to_file(f"{average_runtime}\t")

    average_deviation_from_target = np.mean(deviations_from_target)
    average_energy_control = np.mean(np.array(energies_control))
    if full_solutions:
        average_speedup = np.mean(np.array(full_solution_times) / np.array(solution_times))
        average_error = np.mean(errors)
        average_estimated_error = np.mean(estimated_errors)
        average_estimator_efficiency = np.mean(np.array(estimated_errors) / np.array(errors))
        average_error_control = np.mean(errors_control)
        average_runtime_error_estimator = np.mean(estimation_times)
        average_error_final_state = np.mean(errors_final_time_states)
        average_deviation_from_target_using_fom = np.mean(deviations_from_target_using_fom)
        write_to_file(f"{average_speedup}\t{average_error}\t{average_estimated_error}\t"
                      f"{average_estimator_efficiency}\t{average_error_control}\t{average_energy_control}\t"
                      f"{average_error_final_state}\t{average_deviation_from_target}\t"
                      f"{average_deviation_from_target_using_fom}\t{average_runtime_error_estimator}")
        write_to_test_errors_file("Error\t")
        for err in errors:
            write_to_test_errors_file(f"{err}\t")
        write_to_test_errors_file("\n")
        write_to_test_errors_file("Estimated error\t")
        for est_err in estimated_errors:
            write_to_test_errors_file(f"{est_err}\t")
        write_to_test_errors_file("\n")
        write_to_test_errors_file("Estimated error components\t")
        for est_err_comps in estimated_errors_components:
            write_to_test_errors_file(f"{est_err_comps}\t")
        write_to_test_errors_file("\n")
        write_to_test_errors_file("Error control\t")
        for err in errors_control:
            write_to_test_errors_file(f"{err}\t")
        write_to_test_errors_file("\n")
    else:
        write_to_file(f"\t\t\t\t\t{average_energy_control}\t\t{average_deviation_from_target}\t\t")
        write_to_test_errors_file("Parameter\t")
        for mu in test_parameters:
            write_to_test_errors_file(f"{mu}\t")
        write_to_test_errors_file("\n")
        return solutions, solution_times, controls, final_time_states


def run_reduction_procedures_and_analysis(full_model, logger, tol, inner_tol_double_greedy, max_basis_size,
                                          max_inner_iterations, reduction_strategy, reduction_strategy_parameters,
                                          initial_parameter, training_parameters_state_reduction,
                                          training_parameters_greedy, training_parameters_inner_greedy,
                                          test_parameters):
    problem_name = full_model.title
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename_base = f"results_{problem_name.lower().replace(' ', '_')}_{timestr}"
    pathlib.Path(filename_base).mkdir(parents=True, exist_ok=True)
    filename = f"{filename_base}/summary.txt"

    def write_to_file(line, filename_=filename):
        if filename_:
            with open(filename_, "a") as f:
                f.write(line)

    write_to_file("Problem setting:\n")
    write_to_file(f"\t{problem_name}\n")
    write_to_file(f"System dimension: {full_model.n}\n")
    write_to_file(f"Number of time steps: {full_model.nt}\n")

    write_to_file("Number of training parameters:\n")
    write_to_file(f"\tGreedy: {len(training_parameters_greedy)}\n")
    write_to_file(f"\tState reduction primal: {len(training_parameters_state_reduction)}\n")
    write_to_file(f"\tState reduction adjoint: {len(training_parameters_state_reduction)}\n")
    write_to_file(f"Number of test parameters: {len(test_parameters)}\n")
    write_to_file(f"Tolerance: {tol}\n")
    write_to_file(f"Inner tolerance for double greedy: {inner_tol_double_greedy}\n")
    write_to_file(f"Maximum basis size: {max_basis_size}\n")
    write_to_file(f"Reduction strategy: {reduction_strategy}\n")
    write_to_file(f"Reduction strategy parameters: {reduction_strategy_parameters}\n")

    write_to_file(f"Initial parameter: {initial_parameter}\n")

    write_to_file = partial(write_to_file, filename_=f"{filename_base}/results.txt")
    write_to_test_errors_file = partial(write_to_file, filename_=f"{filename_base}/test_errors.txt")

    def write_offline_timings(offline_timings):
        string = ""
        if offline_timings['required_time_overall']:
            string += str(offline_timings['required_time_overall'])
        string += "\t"
        if offline_timings['required_time_greedy']:
            string += str(offline_timings['required_time_greedy'])
        string += "\t"
        if offline_timings['required_time_system_reductions']:
            string += str(offline_timings['required_time_system_reductions'])
        string += "\t"
        write_to_file(string)

    def write_basis_sizes(model):
        string = ""
        if hasattr(model, 'N'):
            string += str(model.N)
        string += "\t"
        if hasattr(model, 'kpr'):
            string += str(model.kpr)
        string += "\t"
        if hasattr(model, 'kad'):
            string += str(model.kad)
        string += "\t"
        write_to_file(string)

    def write_results_greedy(results_greedy, filename):
        write = partial(write_to_file, filename_=filename)
        tab = "\t"
        error_components_keys = tab.join([str(key) for key in results_greedy["max_error_components"][0].keys()])
        write(f"Iteration\tMaximum estimated error\tMaximum true error\tMaximum efficiency\tMinimum efficiency\t"
              f"{error_components_keys}\t{results_greedy['rom_infos_headline']}\tSelected index\tSelected parameter\n")
        for k, (max_est_err, max_true_err, max_eff, min_eff, max_error_components, rom_infos, sel_ind,
                sel_param) in enumerate(
                zip(results_greedy["max_estimated_errors"], results_greedy["max_true_errors"],
                    results_greedy["max_efficiencies"], results_greedy["min_efficiencies"],
                    results_greedy["max_error_components"], results_greedy["rom_infos"],
                    results_greedy["selected_indices"], results_greedy["selected_parameters"])):
            error_components = tab.join([str(comp) for comp in max_error_components.values()])
            write(f"{k}\t{max_est_err}\t{max_true_err}\t{max_eff}\t{min_eff}\t{error_components}\t{rom_infos}\t"
                  f"{sel_ind}\t{sel_param}\n")

    def write_results_double_greedy(results_greedy, filename):
        write = partial(write_to_file, filename_=filename)
        write("Outer iteration\tInner iteration\tMaximum estimated error\tMaximum true error\tMaximum efficiency\t"
              f"Minimum efficiency\t{results_greedy['rom_infos_headline']}\tSelected index\tSelected parameter\n")
        for k, (max_est_err_list, max_true_err_list, max_eff_list, min_eff_list, rom_infos_list, sel_ind_list,
                sel_param_list) in enumerate(
                zip(results_greedy["max_estimated_errors_inner"], results_greedy["max_true_errors_inner"],
                    results_greedy["max_efficiencies_inner"], results_greedy["min_efficiencies_inner"],
                    results_greedy["rom_infos_inner"], results_greedy["selected_indices_inner"],
                    results_greedy["selected_parameters_inner"])):
            write(f"{k}\n")
            for k_inner, (max_est_err, max_true_err, max_eff, min_eff, rom_infos, sel_ind, sel_param) in enumerate(
                    zip(max_est_err_list, max_true_err_list, max_eff_list, min_eff_list, rom_infos_list, sel_ind_list,
                        sel_param_list)):
                write(f"\t{k_inner}\t{max_est_err}\t{max_true_err}\t{max_eff}\t{min_eff}\t{rom_infos}\t{sel_ind}\t"
                      f"{sel_param}\n")

    write_to_file("Method\tOverall offline time\tOffline time greedy\tOffline time system reductions\tN\tkpr\tkad\t"
                  "Average (online) runtime\tAverage speedup\tAverage error in final time adjoint state\t"
                  "Average estimated error in final time adjoint state\tAverage estimator efficiency\t"
                  "Average error in control\tAverage energy of control\t"
                  "Average error in final state compared to FOM\tAverage deviation from target\t"
                  "Average deviation from target using FOM for simulation\tAverage runtime error estimator\n")

    optimal_adjoints = []
    for mu in training_parameters_greedy:
        optimal_adjoints.append(full_model.solve(mu))

    abbreviation = "FOM"
    with logger.block(abbreviation):
        write_to_file(f"{abbreviation}\t\t\t\t\t\t\t")
        full_solutions, full_solution_times, full_controls, full_final_time_states = \
            run_online_tests(full_model, test_parameters, None, None, None, None, full_model,
                             write_to_file, write_to_test_errors_file)
        write_to_file("\n")

    with logger.block("G-ROM and G-SR-ROM"):
        reduced_model, completely_reduced_model, \
            results_greedy, offline_timings = greedy_on_reduced_model_and_completely_reduced_construction_afterwards(
                    full_model, reduction_strategy, reduction_strategy_parameters, training_parameters_greedy, tol,
                    max_basis_size, training_parameters_state_reduction, training_parameters_state_reduction,
                    return_errors_and_efficiencies=True, optimal_adjoints=optimal_adjoints)
        write_results_greedy(results_greedy, f"{filename_base}/G-SR-ROM.txt")
        abbreviation = "G-ROM"
        write_to_file(f"{abbreviation}\t")
        write_offline_timings(offline_timings)
        write_basis_sizes(reduced_model)
        write_to_test_errors_file(f"{abbreviation}\n")
        run_online_tests(reduced_model, test_parameters, full_solutions, full_solution_times,
                         full_controls, full_final_time_states, full_model,
                         write_to_file, write_to_test_errors_file)
        write_to_file("\n")
        abbreviation = "G-SR-ROM"
        write_to_file(f"{abbreviation}\t")
        write_offline_timings(offline_timings)
        write_basis_sizes(completely_reduced_model)
        write_to_test_errors_file(f"{abbreviation}\n")
        run_online_tests(completely_reduced_model, test_parameters, full_solutions, full_solution_times,
                         full_controls, full_final_time_states, full_model,
                         write_to_file, write_to_test_errors_file)
        write_to_file("\n")

    abbreviation = "GC-ROM"
    with logger.block(abbreviation):
        _, completely_reduced_model, \
            results_greedy, offline_timings = greedy_on_completely_reduced_model(
                    full_model, initial_parameter, reduction_strategy, reduction_strategy_parameters,
                    training_parameters_greedy, tol, max_basis_size,
                    return_errors_and_efficiencies=True, optimal_adjoints=optimal_adjoints)
        write_results_greedy(results_greedy, f"{filename_base}/{abbreviation}.txt")
        write_to_file(f"{abbreviation}\t")
        write_offline_timings(offline_timings)
        write_basis_sizes(completely_reduced_model)
        write_to_test_errors_file(f"{abbreviation}\n")
        run_online_tests(completely_reduced_model, test_parameters, full_solutions, full_solution_times,
                         full_controls, full_final_time_states, full_model,
                         write_to_file, write_to_test_errors_file)
        write_to_file("\n")

    abbreviation = "SR-G-ROM"
    with logger.block(abbreviation):
        _, completely_reduced_model, \
            results_greedy, offline_timings = state_reduction_and_greedy_for_final_time_adjoints_afterwards(
                    full_model, training_parameters_state_reduction, training_parameters_state_reduction,
                    training_parameters_state_reduction, reduction_strategy, reduction_strategy_parameters, tol,
                    max_basis_size, return_errors_and_efficiencies=True)
        write_results_greedy(results_greedy, f"{filename_base}/{abbreviation}.txt")
        write_to_file(f"{abbreviation}\t")
        write_offline_timings(offline_timings)
        write_basis_sizes(completely_reduced_model)
        write_to_test_errors_file(f"{abbreviation}\n")
        run_online_tests(completely_reduced_model, test_parameters, full_solutions, full_solution_times,
                         full_controls, full_final_time_states, full_model,
                         write_to_file, write_to_test_errors_file)
        write_to_file("\n")

    abbreviation = "DG-ROM"
    with logger.block(abbreviation):
        completely_reduced_model, results_greedy, offline_timings = double_greedy_on_completely_reduced_model(
                full_model, initial_parameter, reduction_strategy, reduction_strategy_parameters, tol,
                inner_tol_double_greedy, training_parameters_greedy, training_parameters_inner_greedy, max_basis_size,
                max_inner_iterations, return_errors_and_efficiencies=True, optimal_adjoints=optimal_adjoints)
        write_results_greedy(results_greedy, f"{filename_base}/{abbreviation}.txt")
        write_results_double_greedy(results_greedy, f"{filename_base}/{abbreviation}_inner_greedy.txt")
        write_to_file(f"{abbreviation}\t")
        write_offline_timings(offline_timings)
        write_basis_sizes(completely_reduced_model)
        write_to_test_errors_file(f"{abbreviation}\n")
        run_online_tests(completely_reduced_model, test_parameters, full_solutions, full_solution_times,
                         full_controls, full_final_time_states, full_model,
                         write_to_file, write_to_test_errors_file)
        write_to_file("\n")
