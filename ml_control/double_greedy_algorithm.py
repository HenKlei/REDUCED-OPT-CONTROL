import numpy as np
import time

from ml_control.logger import getLogger


def double_greedy(training_parameters, training_parameters_inner_greedy, fom, c_rom,
                  tol=1e-4, inner_tol=1e-4, max_basis_size=None, max_inner_iterations=None,
                  return_errors_and_efficiencies=False, optimal_adjoints=None):
    """Runs the double greedy algorithm for the given training parameters."""
    if return_errors_and_efficiencies:
        true_errors = []
        efficiencies = []

    c_rom.extend_state_system_basis_on_extend = False

    estimated_errors = []
    estimated_error_components = []

    training_data = []

    logger = getLogger("double_greedy", level="INFO")

    if return_errors_and_efficiencies and optimal_adjoints is None:
        optimal_adjoints = []
        with logger.block("Computing true errors and efficiencies as well! This might be costly!"):
            for i, mu in enumerate(training_parameters):
                phi_opt = fom.solve(mu)
                optimal_adjoints.append(phi_opt)

    logger.info("Select first parameter ...")

    time_efficiencies = 0.
    tic_global = time.perf_counter()

    for i, mu in enumerate(training_parameters):
        phi_red, additional_data = c_rom.solve(mu, return_additional_data=True)
        estimated_error_mu, error_components_mu = c_rom.estimate_error(mu, phi_red, return_error_components=True,
                                                                       **additional_data)

        estimated_errors.append(estimated_error_mu)
        estimated_error_components.append(error_components_mu)

        tic = time.perf_counter()
        if return_errors_and_efficiencies:
            phi_opt = optimal_adjoints[i]
            true_error_mu = fom.spatial_norm(phi_opt - c_rom.reconstruct(phi_red))
            true_errors.append(true_error_mu)
            if not abs(true_error_mu) < 1e-12:
                efficiencies.append(estimated_error_mu / true_error_mu)
        time_efficiencies += time.perf_counter() - tic

    index_max = np.argmax(estimated_errors)
    max_estimated_error = estimated_errors[index_max]
    logger.info(f"Determined first parameter with error {max_estimated_error} ...")
    selected_indices = [index_max]
    max_estimated_errors = [max_estimated_error]
    error_components_keys = estimated_error_components[0].keys()
    max_error_components = [{key: max(item[key] for item in estimated_error_components)
                             for key in error_components_keys}]
    mu = training_parameters[index_max]
    selected_parameters = [mu]
    rom_infos = [c_rom.short_summary()]
    rom_infos_inner_iterations_overall = []
    selected_indices_inner_overall = []
    selected_parameters_inner_overall = []
    max_estimated_inner_errors_overall = []

    tic = time.perf_counter()
    if return_errors_and_efficiencies:
        max_true_errors = [max(true_errors)]
        max_efficiencies = [max(efficiencies, default=1.)]
        min_efficiencies = [min(efficiencies, default=1.)]
        max_true_errors_inner_overall = []
        max_efficiencies_inner_overall = []
        min_efficiencies_inner_overall = []
    time_efficiencies += time.perf_counter() - tic

    logger.info("Starting greedy parameter selection ...")

    k = 0
    while max_estimated_error > tol and (max_basis_size is None or k < max_basis_size):
        logger.info(f"Determined next parameter number {index_max} with error {max_estimated_error} ...")
        selected_indices.append(index_max)

        estimated_errors = []
        estimated_error_components = []
        if return_errors_and_efficiencies:
            true_errors = []
            efficiencies = []

        with logger.block(f"Parameter selection step {k+1}:"):
            logger.info(f"Extending reduced model for selected parameter mu={mu} ...")
            phi_mu = c_rom.extend(mu=mu)
            rom_infos.append(c_rom.short_summary())
            np_vector_array = c_rom.reduced_basis[-1]
            phi_red = c_rom.reduced_basis.inner(np_vector_array, product=c_rom.G_product).flatten()

            with logger.block("Inner greedy:"):
                rom_infos_inner_iterations = [c_rom.short_summary()]
                max_estimated_inner_errors = []
                selected_indices_inner = []
                selected_parameters_inner = []
                if return_errors_and_efficiencies:
                    max_true_errors_inner = []
                    max_efficiencies_inner = []
                    min_efficiencies_inner = []

                estimated_gramian_errors = []
                if return_errors_and_efficiencies:
                    true_errors_inner = []
                    efficiencies_inner = []

                estimated_gramian_errors = []
                for i, mu_inner in enumerate(training_parameters_inner_greedy):
                    estimated_gramian_error = fom.Mnorm * fom.const * c_rom.estimate_gramian_error(mu_inner, phi_red)
                    estimated_gramian_errors.append(estimated_gramian_error)

                    tic = time.perf_counter()
                    if return_errors_and_efficiencies:
                        optimal_gramian_application = fom.compute_gramian_application(mu_inner,
                                                                                      c_rom.reconstruct(phi_red))
                        reduced_gramian_application = c_rom.compute_reduced_gramian_application(mu_inner, phi_red)
                        reconstructed_gramian_application = (c_rom.reconstruct_state([reduced_gramian_application])
                                                             .flatten())
                        true_gramian_error = fom.const * fom.spatial_norm(fom.M @ (optimal_gramian_application
                                                                                   - reconstructed_gramian_application))
                        true_errors_inner.append(true_gramian_error)
                        if not abs(true_error_mu) < 1e-12:
                            efficiencies_inner.append(estimated_gramian_error / true_gramian_error)
                    time_efficiencies += time.perf_counter() - tic

                index_max_inner = np.argmax(estimated_gramian_errors)
                max_estimated_inner_error = estimated_gramian_errors[index_max_inner]
                max_mu_inner = training_parameters_inner_greedy[index_max_inner]
                selected_indices_inner.append(index_max_inner)
                selected_parameters_inner.append(max_mu_inner)
                rom_infos_inner_iterations.append(c_rom.short_summary())
                max_estimated_inner_errors.append(max_estimated_inner_error)
                tic = time.perf_counter()
                if return_errors_and_efficiencies:
                    max_true_errors_inner.append(max(true_errors_inner))
                    max_efficiencies_inner.append(max(efficiencies_inner, default=1.))
                    min_efficiencies_inner.append(min(efficiencies_inner, default=1.))
                time_efficiencies += time.perf_counter() - tic

                k_inner = 0
                while max_estimated_inner_error > inner_tol and (max_inner_iterations is None
                                                                 or k_inner < max_inner_iterations):
                    logger.info(f"Determined next parameter number {index_max_inner} with error "
                                f"{max_estimated_inner_error} ...")
                    logger.info(f"Extending reduced model for selected parameter mu={max_mu_inner} ...")
                    c_rom.extend_state_system_basis(max_mu_inner, phi_mu)

                    estimated_gramian_errors = []
                    if return_errors_and_efficiencies:
                        true_errors_inner = []
                        efficiencies_inner = []

                    with logger.block(f"Parameter selection step {k_inner+1}:"):
                        for i, mu_inner in enumerate(training_parameters_inner_greedy):
                            estimated_gramian_error = fom.Mnorm * fom.const * c_rom.estimate_gramian_error(mu_inner,
                                                                                                           phi_red)
                            estimated_gramian_errors.append(estimated_gramian_error)

                            tic = time.perf_counter()
                            if return_errors_and_efficiencies:
                                optimal_gramian_application = fom.compute_gramian_application(
                                        mu_inner, c_rom.reconstruct(phi_red))
                                reduced_gramian_application = c_rom.compute_reduced_gramian_application(mu_inner,
                                                                                                        phi_red)
                                reconstructed_gramian_application = (
                                        c_rom.reconstruct_state([reduced_gramian_application]).flatten())
                                true_gramian_error = fom.const * fom.spatial_norm(fom.M @ (optimal_gramian_application
                                                                                  - reconstructed_gramian_application))
                                true_errors_inner.append(true_gramian_error)
                                if not abs(true_error_mu) < 1e-12:
                                    efficiencies_inner.append(estimated_gramian_error / true_gramian_error)
                            time_efficiencies += time.perf_counter() - tic

                    index_max_inner = np.argmax(estimated_gramian_errors)
                    max_estimated_inner_error = estimated_gramian_errors[index_max_inner]
                    max_mu_inner = training_parameters_inner_greedy[index_max_inner]
                    logger.info(f"Maximum error in inner iteration {k_inner}: {max_estimated_inner_error} ...")
                    selected_indices_inner.append(index_max_inner)
                    selected_parameters_inner.append(max_mu_inner)
                    rom_infos_inner_iterations.append(c_rom.short_summary())
                    max_estimated_inner_errors.append(max_estimated_inner_error)
                    tic = time.perf_counter()
                    if return_errors_and_efficiencies:
                        max_true_errors_inner.append(max(true_errors_inner))
                        max_efficiencies_inner.append(max(efficiencies_inner, default=1.))
                        min_efficiencies_inner.append(min(efficiencies_inner, default=1.))
                    time_efficiencies += time.perf_counter() - tic
                    k_inner += 1

                rom_infos_inner_iterations_overall.append(rom_infos_inner_iterations)
                max_estimated_inner_errors_overall.append(max_estimated_inner_errors)
                selected_indices_inner_overall.append(selected_indices_inner)
                selected_parameters_inner_overall.append(selected_parameters_inner)
                if return_errors_and_efficiencies:
                    max_true_errors_inner_overall.append(max_true_errors_inner)
                    max_efficiencies_inner_overall.append(max_efficiencies_inner)
                    min_efficiencies_inner_overall.append(min_efficiencies_inner)

            logger.info("Checking errors on training set ...")
            for i, mu in enumerate(training_parameters):
                phi_red, additional_data = c_rom.solve(mu, return_additional_data=True)
                estimated_error_mu, error_components_mu = c_rom.estimate_error(mu, phi_red,
                                                                               return_error_components=True,
                                                                               **additional_data)

                training_data.append((mu, phi_red))
                estimated_errors.append(estimated_error_mu)
                estimated_error_components.append(error_components_mu)

                tic = time.perf_counter()
                if return_errors_and_efficiencies:
                    phi_opt = optimal_adjoints[i]
                    true_error_mu = fom.spatial_norm(phi_opt - c_rom.reconstruct(phi_red))
                    true_errors.append(true_error_mu)
                    if not abs(true_error_mu) < 1e-12:
                        efficiencies.append(estimated_error_mu / true_error_mu)
                time_efficiencies += time.perf_counter() - tic

            index_max = np.argmax(estimated_errors)
            max_estimated_error = estimated_errors[index_max]

            logger.info(f"Maximum estimated error on training set: {max_estimated_error}")
            max_estimated_errors.append(max_estimated_error)
            max_error_components.append({key: max(item[key] for item in estimated_error_components)
                                         for key in error_components_keys})
            selected_indices.append(index_max)
            mu = training_parameters[index_max]
            selected_parameters.append(mu)

            tic = time.perf_counter()
            if return_errors_and_efficiencies:
                max_true_errors.append(max(true_errors))
                max_efficiencies.append(max(efficiencies, default=1.))
                min_efficiencies.append(min(efficiencies, default=1.))
            time_efficiencies += time.perf_counter() - tic
            k += 1

    required_time = time.perf_counter() - tic_global - time_efficiencies

    logger.info("Finished double greedy selection procedure ...")

    results_greedy = {"selected_indices": selected_indices,
                      "selected_indices_inner": selected_indices_inner_overall,
                      "selected_parameters": selected_parameters,
                      "selected_parameters_inner": selected_parameters_inner_overall,
                      "max_estimated_errors": max_estimated_errors,
                      "max_estimated_errors_inner": max_estimated_inner_errors_overall,
                      "max_error_components": max_error_components,
                      "required_time_greedy": required_time,
                      "training_data": training_data,
                      "rom_infos": rom_infos,
                      "rom_infos_headline": c_rom.short_summary_headline(),
                      "rom_infos_inner": rom_infos_inner_iterations_overall}
    if return_errors_and_efficiencies:
        results_greedy["max_true_errors"] = max_true_errors
        results_greedy["max_true_errors_inner"] = max_true_errors_inner_overall
        results_greedy["max_efficiencies"] = max_efficiencies
        results_greedy["min_efficiencies"] = min_efficiencies
        results_greedy["max_efficiencies_inner"] = max_efficiencies_inner_overall
        results_greedy["min_efficiencies_inner"] = min_efficiencies_inner_overall
        results_greedy["optimal_adjoints"] = optimal_adjoints

    return results_greedy
