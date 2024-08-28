import numpy as np
import time

from ml_control.logger import getLogger


def greedy(training_parameters, fom, rom, tol=1e-4, max_basis_size=None, return_errors_and_efficiencies=False,
           optimal_adjoints=None):
    """Runs the greedy algorithm for the given system and training parameters."""
    if return_errors_and_efficiencies:
        true_errors = []
        efficiencies = []

    estimated_errors = []
    estimated_error_components = []

    training_data = []

    logger = getLogger("greedy", level="INFO")

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
        phi_red, additional_data = rom.solve(mu, return_additional_data=True)
        estimated_error_mu, error_components_mu = rom.estimate_error(mu, phi_red, return_error_components=True,
                                                                     **additional_data)

        estimated_errors.append(estimated_error_mu)
        estimated_error_components.append(error_components_mu)

        tic = time.perf_counter()
        if return_errors_and_efficiencies:
            phi_opt = optimal_adjoints[i]
            true_error_mu = fom.spatial_norm(phi_opt - rom.reconstruct(phi_red))
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
    mu_selected = training_parameters[index_max]
    selected_parameters = [mu_selected]
    rom_infos = [rom.short_summary()]

    tic = time.perf_counter()
    if return_errors_and_efficiencies:
        max_true_errors = [max(true_errors)]
        max_efficiencies = [max(efficiencies, default=1.)]
        min_efficiencies = [min(efficiencies, default=1.)]
    time_efficiencies += time.perf_counter() - tic

    logger.info("Starting greedy parameter selection ...")

    k = 0
    while max_estimated_error > tol and (max_basis_size is None or k < max_basis_size):
        logger.info(f"Determined next parameter number {index_max} with error {max_estimated_error} ...")

        estimated_errors = []
        estimated_error_components = []
        if return_errors_and_efficiencies:
            true_errors = []
            efficiencies = []

        with logger.block(f"Parameter selection step {k+1}:"):
            logger.info(f"Extending reduced model for selected parameter mu={mu_selected} ...")
            rom.extend(mu=mu_selected)
            rom_infos.append(rom.short_summary())

            logger.info("Checking errors on training set ...")

            for i, mu in enumerate(training_parameters):
                phi_red, additional_data = rom.solve(mu, return_additional_data=True)
                estimated_error_mu, error_components_mu = rom.estimate_error(mu, phi_red,
                                                                             return_error_components=True,
                                                                             **additional_data)

                training_data.append((mu, phi_red))
                estimated_errors.append(estimated_error_mu)
                estimated_error_components.append(error_components_mu)

                tic = time.perf_counter()
                if return_errors_and_efficiencies:
                    phi_opt = optimal_adjoints[i]
                    true_error_mu = fom.spatial_norm(phi_opt - rom.reconstruct(phi_red))
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
            mu_selected = training_parameters[index_max]
            selected_parameters.append(mu_selected)

            tic = time.perf_counter()
            if return_errors_and_efficiencies:
                max_true_errors.append(max(true_errors))
                max_efficiencies.append(max(efficiencies, default=1.))
                min_efficiencies.append(min(efficiencies, default=1.))
            time_efficiencies += time.perf_counter() - tic
            k += 1

    required_time = time.perf_counter() - tic_global - time_efficiencies

    logger.info("Finished greedy selection procedure ...")

    results_greedy = {"selected_indices": selected_indices,
                      "selected_parameters": selected_parameters,
                      "max_estimated_errors": max_estimated_errors,
                      "max_error_components": max_error_components,
                      "required_time_greedy": required_time,
                      "training_data": training_data,
                      "rom_infos_headline": rom.short_summary_headline(),
                      "rom_infos": rom_infos}
    if return_errors_and_efficiencies:
        results_greedy["max_true_errors"] = max_true_errors
        results_greedy["max_efficiencies"] = max_efficiencies
        results_greedy["min_efficiencies"] = min_efficiencies
        results_greedy["optimal_adjoints"] = optimal_adjoints

    return results_greedy
