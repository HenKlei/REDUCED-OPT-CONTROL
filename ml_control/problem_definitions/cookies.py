import numpy as np
import pathlib
from pymor.tools.io import load_matrix
import os
from scipy.linalg import sqrtm
import scipy.sparse as sps

from ml_control.utilities import Lincomb, LogParameterSpace
from ml_control.full_model import FullModel


def create_cookies_problem(nt, problem_size="medium"):
    T = 1.0

    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, f"cookies_data/{problem_size}/")

    A = [load_matrix(path + f"A{i}.mtx") for i in range(5)]

    parametrized_A = Lincomb([lambda _, t: 1./8. + 14. * (t - 0.25)**2,
                              lambda mu, _: mu[0], lambda mu, _: mu[1]],
                             [A[0], A[1] + A[3], A[2] + A[4]])

    B_matrix = load_matrix(path + "B.txt")[..., np.newaxis]
    B_coefficients = [lambda _, t: 1.]
    parametrized_B = Lincomb(B_coefficients, [B_matrix])

    C = load_matrix(path + "C.txt")

    E = load_matrix(path + "E.mtx")
    n = E.shape[0]

    parametrized_x0 = Lincomb([lambda _: 1.0], [np.zeros(n)])
    parametrized_xT = Lincomb([lambda _: 0.25], [np.ones(n)])

    R = np.eye(1) / 50.
    M = C.T @ C

    parameter_space = LogParameterSpace(1e-1, 1e2, dim=2)

    G = load_matrix(path + "M.mtx")
    C1 = lambda _: 1.

    Q = sps.linalg.spsolve(E, B_matrix @ np.linalg.solve(R, B_matrix.T))
    root = sqrtm(G.todense())
    temp = np.linalg.norm(sps.linalg.spsolve(sps.csr_matrix(root.T), Q.T @ root.T).T, ord=2)
    C2 = lambda _: temp

    const = 1.

    def visualize(x, path_results="cookies_results/", plot_output=True):
        pathlib.Path(path_results).mkdir(parents=True, exist_ok=True)
        # plot state
        import dolfin as df

        mesh = df.Mesh(path + "cookie.xml")
        V = df.FunctionSpace(mesh, "CG", 1)

        if x.shape[1] != V.dim():
            x = x.T
        assert x.shape[1] == V.dim()

        output = df.File(path_results + "out.pvd")
        state = df.Function(V)
        for u in x:
            state.vector()[:] = u
            output << state

        if plot_output:
            # compute and plot output
            nt = len(x)
            out = np.einsum("ij,kj->ki", C, x)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for i in range(4):
                ax.plot(
                    np.linspace(0, T, nt),
                    out[:, i],
                    label=f"Output {i + 1}",
                )
            ax.grid()
            ax.set_xlabel("Time")
            ax.set_ylabel("Temperature")
            ax.legend()
            fig.savefig(path_results + "output.png", dpi=150)
            with open(path_results + "output.txt", "w") as f:
                for to in out:
                    for o in to:
                        f.write(f"{o}\t")
                    f.write("\n")

    return FullModel(T, nt, n, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R, M,
                     parameter_space, mass_matrix=E, C1=C1, C2=C2, const=const, G=G, visualize=visualize,
                     title="Cookie baking problem")
