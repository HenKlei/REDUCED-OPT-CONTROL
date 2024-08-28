import numpy as np
import matplotlib.pyplot as plt

from ml_control.problem_definitions.wave_equation import create_wave_equation_problem_with_two_controls
from ml_control.visualization import plot_final_time_adjoints, plot_controls, plot_final_time_solutions, \
    animate_solution


full_model = create_wave_equation_problem_with_two_controls()

mu = 5.
phiT_opt = full_model.solve(mu)
u_opt = full_model.compute_control(mu, phiT_opt)
x_opt = full_model.compute_state(mu, u_opt)

plot_final_time_adjoints([phiT_opt], title="Final time optimal adjoint")
plot_controls([u_opt], full_model.T, title="Optimal control")
plot_final_time_solutions([x_opt[-1]], title="Final time optimal solution")
animate_solution(x_opt, ylim=(np.min(x_opt) * 1.1, np.max(x_opt) * 1.1), title="Optimal solution")

print(f"Deviation in final time state: {full_model.spatial_norm(x_opt[-1] - full_model.parametrized_xT(mu))}")

fig, axs = plt.subplots(3)
plot_final_time_adjoints([phiT_opt], show_plot=False, ax=axs[0])
axs[0].set_title("Optimal final time adjoint")
plot_controls([u_opt], full_model.T, show_plot=False, ax=axs[1])
axs[1].set_title("Optimal control")
plot_final_time_solutions([x_opt[-1], full_model.parametrized_xT(mu)], labels=["Optimal state", "Target state"],
                          show_plot=False, ax=axs[2])
axs[2].set_title("Final time states")
axs[2].legend()
fig.suptitle(f"Results for parameter {mu}")
plt.show()
