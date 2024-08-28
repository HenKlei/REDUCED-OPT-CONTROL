import numpy as np

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from ml_control.logger import getLogger
from ml_control.systems import solve_homogeneous_system, get_state_from_final_time_adjoint,\
    get_control_from_final_time_adjoint, solve_system


class ReducedModel:
    def __init__(self, reduced_basis, full_model):
        self.reduced_basis = reduced_basis
        if not self.reduced_basis:
            np_vector_space = NumpyVectorSpace(full_model.n)
            self.reduced_basis = np_vector_space.empty()
        self.full_model = full_model
        self.parameter_space = self.full_model.parameter_space

        self.N = len(self.reduced_basis)
        self.n = self.full_model.n
        self.m = self.full_model.m
        self.T = self.full_model.T
        self.nt = self.full_model.nt
        self.dt = self.full_model.T / self.full_model.nt
        self.parametrized_A = self.full_model.parametrized_A
        self.parametrized_B = self.full_model.parametrized_B
        self.parametrized_x0 = self.full_model.parametrized_x0
        self.parametrized_xT = self.full_model.parametrized_xT
        self.mass_matrix = self.full_model.mass_matrix
        self.G = self.full_model.G
        assert self.G.shape == (self.n, self.n)
        self.G_product = NumpyMatrixOperator(self.G)
        self.R_chol = self.full_model.R_chol
        self.M = self.full_model.M
        self.spatial_norm = self.full_model.spatial_norm

        self.logger = getLogger("ReducedModel", level="INFO")

    def summary(self):
        with self.logger.block("Summary:"):
            self.logger.info(f"System dimension: {self.n}")
            self.logger.info(f"Reduced basis size N (final time optimal adjoint): {self.N}")

    def short_summary_headline(self):
        return "N"

    def short_summary(self):
        return f"{self.N}"

    def solve(self, mu, return_additional_data=False):
        """Solves the reduced basis reduced model for the given parameter."""
        if self.reduced_basis is not None and len(self.reduced_basis) > 0:
            A = self.parametrized_A(mu)
            B = self.parametrized_B(mu)
            x0 = self.parametrized_x0(mu)
            xT = self.parametrized_xT(mu)

            x0_T_mu = solve_homogeneous_system(x0, self.T, self.nt, A, mass_matrix=self.mass_matrix)[-1]

            mat = np.array([self.mass_matrix.T @ phi.to_numpy().flatten()
                            - self.M @ get_state_from_final_time_adjoint(phi.to_numpy().flatten(), np.zeros(self.n),
                                                                         self.T, self.nt, A, B, self.R_chol,
                                                                         mass_matrix=self.mass_matrix)[-1]
                            for phi in self.reduced_basis]).T
            phi_red = np.linalg.solve(mat.T @ self.G @ mat,
                                      mat.T @ self.G @ self.M @ (x0_T_mu - xT))
        else:
            mat = None
            phi_red = np.zeros(0)
        if return_additional_data:
            return phi_red, {"projection_matrix": mat}
        return phi_red

    def reconstruct(self, phi_red):
        if self.reduced_basis is not None and len(self.reduced_basis) > 0:
            return self.reduced_basis.lincomb(phi_red).to_numpy().flatten()
        return np.zeros(self.n)

    def reconstruct_state(self, x):
        return x

    def compute_control(self, mu, phi_red):
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        return get_control_from_final_time_adjoint(self.reconstruct(phi_red), self.T, self.nt, A, B, self.R_chol,
                                                   mass_matrix=self.mass_matrix)

    def compute_state(self, mu, u):
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        x0 = self.parametrized_x0(mu)
        return solve_system(x0, self.T, self.nt, A, B, u, mass_matrix=self.mass_matrix)

    def estimate_error(self, mu, phi_red, return_error_components=False, projection_matrix=None):
        """Estimates the error in the final time adjoint."""
        A = self.parametrized_A(mu)
        B = self.parametrized_B(mu)
        x0 = self.parametrized_x0(mu)
        xT = self.parametrized_xT(mu)
        x0_T_mu = solve_homogeneous_system(x0, self.T, self.nt, A, mass_matrix=self.mass_matrix)[-1]

        if self.reduced_basis is not None and len(self.reduced_basis) > 0:
            if projection_matrix is not None:
                projection = projection_matrix @ phi_red
            else:
                phi_reduced = self.reduced_basis.lincomb(phi_red).to_numpy().flatten()
                projection = (self.mass_matrix.T @ phi_reduced
                              - self.M @ get_state_from_final_time_adjoint(phi_reduced, np.zeros(self.n), self.T,
                                                                           self.nt, A, B, self.R_chol,
                                                                           mass_matrix=self.mass_matrix)[-1])
        else:
            projection = np.zeros(self.n)

        diff = projection - self.M @ (x0_T_mu - xT)
        err = self.full_model.compute_dual_weighted_norm(diff)
        err *= self.full_model.const
        if return_error_components:
            return err, {}
        return err

    def extend(self, mu=None, phi=None):
        assert (mu is not None) or (phi is not None)

        if phi is None:
            phi = self.full_model.solve(mu)

        # Orthonormalization of basis by Gram-Schmidt algorithm
        phi_temp = phi.copy()
        for p in self.reduced_basis:
            phi_temp -= self.full_model.G_dot(p.to_numpy().flatten(), phi) * p.to_numpy().flatten()
        if np.isclose(self.full_model.G_norm(phi_temp)**2, 0.):
            self.logger.warn("Not extending reduced basis since new basis vector would be linearly dependent!")
            return phi
        phi_temp = phi_temp / self.full_model.G_norm(phi_temp)
        np_vector_space = NumpyVectorSpace(self.n)
        self.reduced_basis.append(np_vector_space.from_numpy(phi_temp))
        self.N += 1
        assert self.N == len(self.reduced_basis)

        return phi
