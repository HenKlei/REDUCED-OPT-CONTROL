import numpy as np
import pathlib
import matplotlib.pyplot as plt

from ml_control.problem_definitions.wave_equation import create_wave_equation_problem


k_train = 50
damping_forces = [0, 1, 5, 10, 50, 100]

svs = []

for damping in damping_forces:
    full_model = create_wave_equation_problem(damping_force=damping)
    training_parameters = np.linspace(*full_model.parameter_space, k_train)

    optimal_adjoints = []
    for mu in training_parameters:
        phi_opt = full_model.solve(mu)
        optimal_adjoints.append(phi_opt)
    _, singular_values, _ = np.linalg.svd(np.array(optimal_adjoints))
    plt.semilogy(singular_values, label=f"Damping force={damping}")
    svs.append(singular_values)

plt.legend()
plt.show()

write_results = True
if write_results:
    filepath_prefix = 'plot_data_undamped/'
    pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)
    with open(filepath_prefix + 'singular_values_optimal_adjoints_comparison.txt', 'w') as f:
        f.write("Damping force:\t")
        for damping in damping_forces:
            f.write(f"{damping}\t")
        f.write("\n")
        for i in range(len(svs[0])):
            f.write(f"{i + 1}\t")
            for j in range(len(damping_forces)):
                f.write(f"{svs[j][i]}\t")
            f.write("\n")
