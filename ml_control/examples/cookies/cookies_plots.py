from ml_control.problem_definitions.cookies import create_cookies_problem
from ml_control.visualization import plot_controls


nt = 50
full_model = create_cookies_problem(nt, problem_size="medium")

mu = [100., 0.1]

phi_opt = full_model.solve(mu)
u_opt = full_model.compute_control(mu, phi_opt)
plot_controls([u_opt], full_model.T, title="Optimal control")
x_opt = full_model.compute_state(mu, u_opt)

full_model.visualize(x_opt)

full_model.visualize(phi_opt.reshape(-1, 1), path_results="cookies_results/adjoint/", plot_output=False)

with open("cookies_results/optimal_control.txt", "w") as f:
    for u in u_opt.squeeze():
        f.write(f"{u}\n")
