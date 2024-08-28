import numpy as np
from scipy.linalg import cho_factor, sqrtm
import scipy.sparse as sps

from pymor.operators.numpy import NumpyMatrixOperator

from ml_control.systems import get_control_from_final_time_adjoint, solve_optimal_control_problem, solve_system,\
        solve_adjoint
from ml_control.visualization import animate_solution


class FullModel:
    def __init__(self, T, nt, n, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R, M,
                 parameter_space, mass_matrix, solver_options={}, temporal_norm=None,
                 C1=lambda _: 1., C2=lambda _: 1., const=1., G=None, visualize=None, title=""):
        self.T = T
        self.nt = nt
        self.n = n
        self.parametrized_A = parametrized_A
        self.parametrized_B = parametrized_B
        self.m = self.parametrized_B.shape[1]
        assert self.parametrized_B.shape[0] == n
        assert self.parametrized_A.shape == (n, n)
        self.parametrized_x0 = parametrized_x0
        assert self.parametrized_x0.shape[0] == n
        self.parametrized_xT = parametrized_xT
        assert self.parametrized_xT.shape[0] == n
        self.R = R
        self.R_chol = cho_factor(R)
        self.M = M
        if G is None:
            G = np.eye(self.n)
        self.G = G
        if sps.issparse(self.G):
            c = np.abs(np.abs(self.G - self.G.T) - 1e-5 * np.abs(self.G.T))
            assert c.max() <= 1e-8
        else:
            assert np.allclose(self.G, self.G.T)
        self.G_dot = lambda x, y: x @ self.G @ y
        self.G_norm = lambda x: np.sqrt(self.G_dot(x, x))
        self.G_product = NumpyMatrixOperator(self.G)
        self.Mnorm = 1.
        if M is not None:
            if sps.issparse(self.G):
                root = sqrtm(self.G.todense())
            else:
                root = sqrtm(self.G)
            self.Mnorm = np.linalg.norm(sps.linalg.spsolve(sps.csr_matrix(root.T), M.T @ root.T).T, ord=2)
        assert self.M.shape == (n, n)
        self.parameter_space = parameter_space
        self.solver_options = solver_options
        if temporal_norm:
            self.temporal_norm = temporal_norm
        else:
            self.temporal_norm = lambda u: np.sqrt(np.sum(np.linalg.norm(u, axis=-1)**2) * (T / nt))
        self.C1 = C1
        self.C2 = C2
        self.const = const
        self.spatial_norm = self.G_norm
        self.visualize = visualize if visualize is not None else animate_solution
        self.title = title
        self.mass_matrix = mass_matrix
        self.time_varying = self.parametrized_A.time_dependent or self.parametrized_B.time_dependent

    def solve(self, mu, phiT_init=None):
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        x0 = self.parametrized_x0(mu)
        xT = self.parametrized_xT(mu)
        if phiT_init is None:
            phiT_init = np.zeros(self.n)
        return solve_optimal_control_problem(x0, xT, self.T, self.nt, A, B, self.R_chol, self.M, phiT_init,
                                             mass_matrix=self.mass_matrix, **self.solver_options)

    def compute_control(self, mu, phiT):
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        return get_control_from_final_time_adjoint(phiT, self.T, self.nt, A, B, self.R_chol,
                                                   mass_matrix=self.mass_matrix)

    def compute_state(self, mu, u):
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        x0 = self.parametrized_x0(mu)
        return solve_system(x0, self.T, self.nt, A, B, u, mass_matrix=self.mass_matrix)

    def compute_adjoint(self, mu, phiT):
        A = self.parametrized_A(mu)
        return solve_adjoint(phiT, self.T, self.nt, A, mass_matrix=self.mass_matrix)

    def compute_gramian_application(self, mu, phiT):
        u = self.compute_control(mu, phiT)
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        x0_zeros = np.zeros(self.n)
        return - solve_system(x0_zeros, self.T, self.nt, A, B, u, mass_matrix=self.mass_matrix)[-1]

    def deviation_from_target(self, xT, mu):
        target = self.parametrized_xT(mu)
        return (xT - target).T @ self.M @ (xT - target)

    def control_energy(self, u):
        return np.sum(np.linalg.norm(np.einsum('...k,km,...m->...', u, self.R, u), axis=-1)) * (self.T / self.nt)

    def compute_weighted_norm(self, x):
        return self.G_norm(x)

    def compute_dual_weighted_norm(self, x):
        if sps.issparse(self.G):
            return np.sqrt(x.T @ sps.linalg.spsolve(self.G, x))
        return np.sqrt(x.T @ np.linalg.solve(self.G, x))

    def reconstruct_state(self, x):
        return x

    def _summary(self, func, postfix=""):
        func(f"Title: {self.title}" + postfix)
        func(f"Number of time steps: {self.nt}" + postfix)
        func(f"State dimension: {self.n}" + postfix)
