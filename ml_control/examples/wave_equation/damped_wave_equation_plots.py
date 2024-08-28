import numpy as np
import matplotlib.pyplot as plt
import pathlib

from ml_control.problem_definitions.wave_equation import create_wave_equation_problem
from ml_control.visualization import plot_final_time_adjoints, plot_controls, plot_final_time_solutions, \
    animate_solution


full_model = create_wave_equation_problem(damping_force=10.)

mu = 5.
phiT_opt = full_model.solve(mu)

u_opt = full_model.compute_control(mu, phiT_opt)
x_opt = full_model.compute_state(mu, u_opt)

final_time_solution = np.hstack([x_opt[-1], u_opt[-1]])

plot_final_time_adjoints([phiT_opt], title="Final time optimal adjoint")
plot_controls([u_opt], full_model.T, title="Optimal control")
plot_final_time_solutions([final_time_solution], title="Final time optimal solution")
animate_solution(x_opt[:, :full_model.n // 2], ylim=(np.min(x_opt) * 1.1, np.max(x_opt) * 1.1),
                 title="Optimal solution")

print(f"Deviation in final time state: {full_model.spatial_norm(x_opt[-1] - full_model.parametrized_xT(mu))}")

fig, axs = plt.subplots(3)
plot_final_time_adjoints([phiT_opt], show_plot=False, ax=axs[0])
axs[0].set_title("Optimal final time adjoint")
plot_controls([u_opt], full_model.T, show_plot=False, ax=axs[1])
axs[1].set_title("Optimal control")
plot_final_time_solutions([full_model.parametrized_x0(mu), x_opt[-1], full_model.parametrized_xT(mu)],
                          labels=["Initial state", "Optimal state", "Target state"], show_plot=False, ax=axs[2])
axs[2].set_title("Final time states")
axs[2].legend()
fig.suptitle(f"Results for parameter {mu}")
plt.show()

write_results = True
if write_results:
    filepath_prefix = 'plot_data_damped/'
    pathlib.Path(filepath_prefix).mkdir(parents=True, exist_ok=True)
    h = 1. / (full_model.n // 2 + 1)
    with open(filepath_prefix + 'optimal_final_time_adjoint.txt', 'w') as f:
        for x, p in zip(np.linspace(h, 1.-h, phiT_opt.shape[0]), phiT_opt):
            f.write(f'{x}\t{p}\n')
    with open(filepath_prefix + 'optimal_control.txt', 'w') as f:
        for x, u in zip(np.linspace(0, full_model.T, u_opt.shape[0]), u_opt):
            f.write(f'{x}')
            for u_comp in u:
                f.write(f'\t{u_comp}')
            f.write('\n')
    with open(filepath_prefix + 'final_time_state.txt', 'w') as f:
        f.write('0\t0\t0\t0\n')
        for x, state, initial, target in zip(np.linspace(h, 1.-h, full_model.n), x_opt[-1],
                                             full_model.parametrized_x0(mu), full_model.parametrized_xT(mu)):
            f.write(f'{x}\t{state}\t{initial}\t{target}\n')
        f.write(f'0.5\t{u_opt[-1, 0]}\t0\t1.0\n')
    with open(filepath_prefix + 'optimal_state_trajectory.txt', 'w') as f:
        x_opt = np.hstack([np.zeros((x_opt.shape[0], 1)), x_opt, u_opt[:, 0][:, np.newaxis]])
        coords = np.meshgrid(np.linspace(0, full_model.T, x_opt.shape[0]), np.linspace(0, 1, x_opt.shape[1]))
        num_x = 5
        num_t = 50
        x_opt_truncated = x_opt[:, :(full_model.n // 2 + 2)]
        for x, y, z in zip(np.vstack([coords[0][::num_x], coords[0][-1]])[:, ::num_t].flatten(),
                           np.vstack([coords[1][::num_x], coords[1][-1]])[:, ::num_t].flatten(),
                           np.vstack([x_opt_truncated.T[::num_x],
                                      x_opt_truncated.T[full_model.n // 2]])[:, ::num_t].flatten()):
            f.write(f'{x}\t{y}\t{z}\n')
