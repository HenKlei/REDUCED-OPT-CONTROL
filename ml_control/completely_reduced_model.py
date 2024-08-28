from contextlib import nullcontext
import itertools
import numpy as np
import scipy.sparse as sps

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.hapod import dist_vectorarray_hapod, inc_vectorarray_hapod
from pymor.vectorarrays.numpy import NumpyVectorSpace

from ml_control.logger import getLogger
from ml_control.reduced_model import ReducedModel
from ml_control.systems import solve_homogeneous_system, solve_adjoint, get_control_from_adjoint, solve_system,\
        get_control_from_final_time_adjoint
from ml_control.utilities import Lincomb


class CompletelyReducedModel(ReducedModel):
    def __init__(self, Vpr, Wpr, Vad, Wad, reduced_basis, full_model,
                 reduction_strategy=None, reduction_strategy_parameters={},
                 log_detailed_errors=True, extend_state_system_basis_on_extend=True, svals_pr=None, svals_ad=None):
        super().__init__(reduced_basis, full_model)

        self.Vpr = Vpr
        self.Wpr = Wpr
        self.Vad = Vad
        self.Wad = Wad
        self.svals_pr = svals_pr
        self.svals_ad = svals_ad
        assert isinstance(self.full_model.parametrized_A, Lincomb)
        assert self.full_model.parametrized_A.shape == (self.n, self.n)
        assert isinstance(self.full_model.parametrized_B, Lincomb)
        assert self.full_model.parametrized_B.shape == (self.n, self.m)
        assert isinstance(self.full_model.parametrized_x0, Lincomb)
        assert self.full_model.parametrized_x0.shape == (self.n, )
        assert isinstance(self.full_model.parametrized_xT, Lincomb)
        assert self.full_model.parametrized_xT.shape == (self.n, )
        self.C1 = full_model.C1
        self.C2 = full_model.C2
        self.time_varying = self.full_model.time_varying

        self.compute_weighted_norm = full_model.compute_weighted_norm
        self.compute_dual_weighted_norm = full_model.compute_dual_weighted_norm

        self.reduction_strategy = reduction_strategy
        self.reduction_strategy_parameters = reduction_strategy_parameters

        self.extend_state_system_basis_on_extend = extend_state_system_basis_on_extend

        self.log_detailed_errors = log_detailed_errors
        self.logger = getLogger("CompletelyReducedModel", level="INFO")

        self._check_biorthogonality()

        self.precompute()

    def summary(self):
        with self.logger.block("Summary:"):
            self.logger.info(f"System dimension: {self.n}")
            self.logger.info(f"Reduced basis size N (final time optimal adjoint): {self.N}")
            self.logger.info(f"Reduced basis size k_pr (primal system trajectories): {self.Vpr.shape[1]}")
            self.logger.info(f"Reduced basis size k_ad (adjoint system trajectories): {self.Vad.shape[1]}")

    def short_summary_headline(self):
        return "N\tk_pr\tk_ad"

    def short_summary(self):
        return f"{self.N}\t{self.Vpr.shape[1]}\t{self.Vad.shape[1]}"

    def precompute(self):
        Vpr = self.Vpr
        Wpr = self.Wpr
        Vad = self.Vad
        Wad = self.Wad
        if len(self.reduced_basis) == 0:
            Vtilde = np.zeros((self.n, 0))
        else:
            Vtilde = self.reduced_basis.to_numpy().T

        def is_positive_definite(X):
            if sps.issparse(X):
                eigvals, _ = sps.linalg.eigsh(X, k=1, which='SA')
                return (abs(X - X.T) > 1e-10).nnz == 0 and eigvals[0] > 0
            return np.allclose(X, X.T) and np.all(np.linalg.eigvalsh(X) > 0)

        def is_positive_semidefinite(X):
            if sps.issparse(X):
                eigvals, _ = sps.linalg.eigsh(X, k=1, which='SA')
                return (abs(X - X.T) > 1e-10).nnz == 0 and (eigvals[0] >= 0 or np.isclose(eigvals[0], 0.))
            eigvals = np.linalg.eigvals(X)
            return np.allclose(X, X.T) and np.all(eigvals[~np.isclose(eigvals, 0.)] >= 0.)

        assert is_positive_definite(self.G)
        M = self.full_model.M
        assert is_positive_semidefinite(M)

        mass = self.full_model.mass_matrix

        n = Vpr.shape[0]
        kpr = Vpr.shape[1]
        self.kpr = kpr
        assert Wpr.shape[1] == kpr
        kad = Vad.shape[1]
        self.kad = kad
        assert Wad.shape[1] == kad
        N = self.N
        assert N == Vtilde.shape[1]

        assert np.linalg.matrix_rank(Vpr) == kpr
        assert np.linalg.matrix_rank(Wpr) == kpr
        assert np.linalg.matrix_rank(Vad) == kad
        assert np.linalg.matrix_rank(Wad) == kad
        if N > 0:
            assert np.linalg.matrix_rank(Vtilde) == N

        m = self.full_model.parametrized_B.shape[1]

        self.parametrized_A_pr = Lincomb(self.full_model.parametrized_A.parameter_functions,
                                         [Wpr.T @ X @ Vpr for X in self.full_model.parametrized_A.operators])
        assert self.parametrized_A_pr.shape == (kpr, kpr)
        self.parametrized_A_ad = Lincomb(self.full_model.parametrized_A.parameter_functions,
                                         [Vad.T @ X @ Wad for X in self.full_model.parametrized_A.operators])
        assert self.parametrized_A_ad.shape == (kad, kad)
        self.parametrized_B_pr = Lincomb(self.full_model.parametrized_B.parameter_functions,
                                         [Wpr.T @ X for X in self.full_model.parametrized_B.operators])
        assert self.parametrized_B_pr.shape == (kpr, m)
        self.parametrized_B_ad = Lincomb(self.full_model.parametrized_B.parameter_functions,
                                         [Vad.T @ X for X in self.full_model.parametrized_B.operators])
        assert self.parametrized_B_ad.shape == (kad, m)
        self.parametrized_x0 = Lincomb(self.full_model.parametrized_x0.parameter_functions,
                                       [Vpr.T @ self.G @ x for x in self.full_model.parametrized_x0.operators])
        assert self.parametrized_x0.shape == (kpr, )
        if mass is not None:
            self.mass_matrix_pr = Wpr.T @ mass @ Vpr
            self.mass_matrix_ad = Wad.T @ mass @ Vad
        else:
            self.mass_matrix_pr = None
            self.mass_matrix_ad = None

        A_premultiplied_operators = [sps.linalg.spsolve(mass, X).reshape(X.shape)
                                     for X in self.full_model.parametrized_A.operators]
        B_premultiplied_operators = [sps.linalg.spsolve(mass, X).reshape(X.shape)
                                     for X in self.full_model.parametrized_B.operators]

        # Components for primal estimator
        if self.time_varying:
            temp_params = [(lambda y: lambda x: lambda mu, t: x(mu, t) * y(mu, t))(y)(x)
                           for x, y in itertools.product(self.full_model.parametrized_A.parameter_functions,
                                                         self.full_model.parametrized_A.parameter_functions)]
        else:
            temp_params = [(lambda y: lambda x: lambda mu: x(mu) * y(mu))(y)(x)
                           for x, y in itertools.product(self.full_model.parametrized_A.parameter_functions,
                                                         self.full_model.parametrized_A.parameter_functions)]
        temp_operators = [Vpr.T @ X.T @ self.G @ Y @ Vpr
                          for X, Y in itertools.product(A_premultiplied_operators,
                                                        A_premultiplied_operators)]
        self.M1 = Lincomb(temp_params, temp_operators)
        assert self.M1.shape == (kpr, kpr)
        if self.time_varying:
            temp_params = [(lambda y: lambda x: lambda mu, t: x(mu, t) * y(mu, t))(y)(x)
                           for x, y in itertools.product(self.full_model.parametrized_B.parameter_functions,
                                                         self.full_model.parametrized_B.parameter_functions)]
        else:
            temp_params = [(lambda y: lambda x: lambda mu: x(mu) * y(mu))(y)(x)
                           for x, y in itertools.product(self.full_model.parametrized_B.parameter_functions,
                                                         self.full_model.parametrized_B.parameter_functions)]
        temp_operators = [X.T @ self.G @ Y
                          for X, Y in itertools.product(B_premultiplied_operators,
                                                        B_premultiplied_operators)]
        self.M2 = Lincomb(temp_params, temp_operators)
        assert self.M2.shape == (m, m)
        self.M3 = Vpr.T @ self.G @ Vpr
        assert self.M3.shape == (kpr, kpr)
        assert is_positive_semidefinite(self.M3)
        if self.time_varying:
            temp_params = [(lambda y: lambda x: lambda mu, t: x(mu, t) * y(mu, t))(y)(x)
                           for x, y in itertools.product(self.full_model.parametrized_B.parameter_functions,
                                                         self.full_model.parametrized_A.parameter_functions)]
        else:
            temp_params = [(lambda y: lambda x: lambda mu: x(mu) * y(mu))(y)(x)
                           for x, y in itertools.product(self.full_model.parametrized_B.parameter_functions,
                                                         self.full_model.parametrized_A.parameter_functions)]
        temp_operators = [X.T @ self.G @ Y @ Vpr
                          for X, Y in itertools.product(B_premultiplied_operators,
                                                        A_premultiplied_operators)]
        self.M4 = Lincomb(temp_params, temp_operators)
        assert self.M4.shape == (m, kpr)
        temp_params = self.full_model.parametrized_A.parameter_functions
        temp_operators = [Vpr.T @ self.G @ X @ Vpr
                          for X in A_premultiplied_operators]
        self.M5 = Lincomb(temp_params, temp_operators)
        assert self.M5.shape == (kpr, kpr)
        temp_params = self.full_model.parametrized_B.parameter_functions
        temp_operators = [Vpr.T @ self.G @ X
                          for X in B_premultiplied_operators]
        self.M6 = Lincomb(temp_params, temp_operators)
        assert self.M6.shape == (kpr, m)

        # Components for adjoint estimator
        if self.time_varying:
            temp_params = [(lambda y: lambda x: lambda mu, t: x(mu, t) * y(mu, t))(y)(x)
                           for x, y in itertools.product(self.full_model.parametrized_A.parameter_functions,
                                                         self.full_model.parametrized_A.parameter_functions)]
        else:
            temp_params = [(lambda y: lambda x: lambda mu: x(mu) * y(mu))(y)(x)
                           for x, y in itertools.product(self.full_model.parametrized_A.parameter_functions,
                                                         self.full_model.parametrized_A.parameter_functions)]
        temp_operators = [Vad.T @ X.T.T @ self.G @ Y.T @ Vad
                          for X, Y in itertools.product(self.full_model.parametrized_A.operators,
                                                        self.full_model.parametrized_A.operators)]
        self.M7 = Lincomb(temp_params, temp_operators)
        assert self.M7.shape == (kad, kad)
        self.M8 = Vad.T @ mass.T @ self.G @ mass @ Vad
        assert self.M8.shape == (kad, kad)
        assert is_positive_semidefinite(self.M8)
        temp_params = self.full_model.parametrized_A.parameter_functions
        temp_operators = [Vad.T @ mass.T @ self.G @ X.T @ Vad
                          for X in self.full_model.parametrized_A.operators]
        self.M9 = Lincomb(temp_params, temp_operators)
        assert self.M9.shape == (kad, kad)

        temp_mat = Vtilde - Vad @ Vad.T @ self.G @ Vtilde
        self.M10 = temp_mat.T @ self.G @ temp_mat
        assert self.M10.shape == (N, N)

        temp_params = [(lambda y: lambda x: lambda mu: x(mu) * y(mu))(y)(x)
                       for x, y in itertools.product(self.full_model.parametrized_x0.parameter_functions,
                                                     self.full_model.parametrized_x0.parameter_functions)]
        temp_mat = np.eye(n) - Vpr @ Vpr.T @ self.G
        temp_operators = [x.T @ temp_mat.T @ self.G @ temp_mat @ y
                          for x, y in itertools.product(self.full_model.parametrized_x0.operators,
                                                        self.full_model.parametrized_x0.operators)]
        self.eprim0squared = Lincomb(temp_params, temp_operators)

        # Components for reduced error estimator
        self.M11 = Vad.T @ self.G @ Vtilde
        assert self.M11.shape == (kad, N)

        self.M12 = Vtilde.T @ mass @ sps.linalg.spsolve(self.G, mass.T @ Vtilde).reshape((n, N))
        assert self.M12.shape == (N, N)
        assert is_positive_definite(self.M12)

        self.M13 = Vtilde.T @ mass @ sps.linalg.spsolve(self.G, M @ Vpr).reshape((n, kpr))
        assert self.M13.shape == (N, kpr)

        temp_params = self.full_model.parametrized_x0.parameter_functions
        temp_operators = [Vpr.T @ self.G @ x for x in self.full_model.parametrized_x0.operators]
        self.Vtopx0 = Lincomb(temp_params, temp_operators)
        assert self.Vtopx0.shape == (kpr, )

        temp_params = self.full_model.parametrized_xT.parameter_functions
        temp_operators = [Vpr.T @ M.T @ sps.linalg.spsolve(self.G, M @ x)
                          for x in self.full_model.parametrized_xT.operators]
        self.VtopMtopGMxT = Lincomb(temp_params, temp_operators)
        assert self.VtopMtopGMxT.shape == (kpr, )

        temp_params = self.full_model.parametrized_xT.parameter_functions
        temp_operators = [Vtilde.T @ mass @ sps.linalg.spsolve(self.G, M @ x)
                          for x in self.full_model.parametrized_xT.operators]
        self.VtopGMxT = Lincomb(temp_params, temp_operators)
        assert self.VtopGMxT.shape == (N, )

        temp_params = [(lambda y: lambda x: lambda mu: x(mu) * y(mu))(y)(x)
                       for x, y in itertools.product(self.full_model.parametrized_xT.parameter_functions,
                                                     self.full_model.parametrized_xT.parameter_functions)]
        temp_operators = [x.T @ M.T @ sps.linalg.spsolve(self.G, M @ y)
                          for x, y in itertools.product(self.full_model.parametrized_xT.operators,
                                                        self.full_model.parametrized_xT.operators)]
        self.xTtopMtopGMxT = Lincomb(temp_params, temp_operators)

        self.M14 = Vtilde.T @ mass @ sps.linalg.spsolve(self.G, M @ Vpr).reshape((n, kpr))
        assert self.M14.shape == (N, kpr)

        self.M15 = Vpr.T @ M.T @ sps.linalg.spsolve(self.G, M @ Vpr).reshape((n, kpr))
        assert self.M15.shape == (kpr, kpr)
        assert is_positive_semidefinite(self.M15)

    def _norm_res_prim(self, mu, x_hat_s, d_x_hat_s, u_s):
        return (x_hat_s.T @ self.M1(mu) @ x_hat_s + u_s.T @ self.M2(mu) @ u_s
                + d_x_hat_s.T @ self.M3 @ d_x_hat_s + 2 * u_s.T @ self.M4(mu) @ x_hat_s
                - 2 * d_x_hat_s.T @ self.M5(mu) @ x_hat_s - 2 * d_x_hat_s.T @ self.M6(mu) @ u_s)

    def _norm_res_prim_time_dep(self, mu, x_hat_s, d_x_hat_s, u_s, s):
        return (x_hat_s.T @ self.M1(mu, s) @ x_hat_s + u_s.T @ self.M2(mu, s) @ u_s
                + d_x_hat_s.T @ self.M3 @ d_x_hat_s + 2 * u_s.T @ self.M4(mu, s) @ x_hat_s
                - 2 * d_x_hat_s.T @ self.M5(mu, s) @ x_hat_s - 2 * d_x_hat_s.T @ self.M6(mu, s) @ u_s)

    def delta_prim(self, mu, x_hat, d_x_hat, u, dt):
        err_prim_0 = np.sqrt(np.abs(self.eprim0squared(mu)))
        if callable(self.M1(mu)):
            int_norm_res_prim = self.dt * np.sum(np.sqrt(np.abs([self._norm_res_prim_time_dep(mu, x_hat_s, d_x_hat_s,
                                                                                              u_s, (i+1) * dt)
                                                                 for i, (x_hat_s, d_x_hat_s, u_s)
                                                                 in enumerate(zip(x_hat[1:], d_x_hat, u[1:]))])))
        else:
            int_norm_res_prim = self.dt * np.sum(np.sqrt(np.abs([self._norm_res_prim(mu, x_hat_s, d_x_hat_s, u_s)
                                                                 for x_hat_s, d_x_hat_s, u_s
                                                                 in zip(x_hat[1:], d_x_hat, u[1:])])))
        return self.C1(mu) * (err_prim_0 + int_norm_res_prim)

    def _norm_res_adjo(self, mu, phi_hat_s, d_phi_hat_s):
        return (phi_hat_s.T @ self.M7(mu) @ phi_hat_s + d_phi_hat_s.T @ self.M8 @ d_phi_hat_s
                + 2. * d_phi_hat_s.T @ self.M9(mu) @ phi_hat_s)

    def _norm_res_adjo_time_dep(self, mu, phi_hat_s, d_phi_hat_s, s):
        return (phi_hat_s.T @ self.M7(mu, s) @ phi_hat_s + d_phi_hat_s.T @ self.M8 @ d_phi_hat_s
                + 2. * d_phi_hat_s.T @ self.M9(mu, s) @ phi_hat_s)

    def delta_lambda(self, mu, phi_N, x_hat, d_x_hat, u_hat, phi_hat, d_phi_hat, dt):
        err_adjo_T = np.sqrt(np.abs(phi_N.T @ self.M10 @ phi_N))
        if callable(self.M7(mu)):
            int_adjo = self.dt * np.sum(np.cumsum(np.sqrt(np.abs([self._norm_res_adjo_time_dep(mu, phi_hat_s,
                                                                                               d_phi_hat_s, i * dt)
                                                                  for i, (phi_hat_s, d_phi_hat_s)
                                                                  in enumerate(zip(phi_hat[:-1], d_phi_hat))])),
                                                  axis=-1))
            int_adjo += err_adjo_T
            int_adjo *= self.C1(mu)
            int_norm_res_prim = self.dt * np.sum(np.sqrt(np.abs([self._norm_res_prim_time_dep(mu, x_hat_s, d_x_hat_s,
                                                                                              u_hat_s, (i+1) * dt)
                                                                 for i, (x_hat_s, d_x_hat_s, u_hat_s)
                                                                 in enumerate(zip(x_hat[1:], d_x_hat,
                                                                                  u_hat[1:]))])))
        else:
            int_adjo = self.dt * np.sum(np.cumsum(np.sqrt(np.abs([self._norm_res_adjo(mu, phi_hat_s, d_phi_hat_s)
                                                                  for phi_hat_s, d_phi_hat_s
                                                                  in zip(phi_hat[:-1], d_phi_hat)])),
                                                  axis=-1))
            int_adjo += err_adjo_T
            int_adjo *= self.C1(mu)
            int_norm_res_prim = self.dt * np.sum(np.sqrt(np.abs([self._norm_res_prim(mu, x_hat_s, d_x_hat_s, u_hat_s)
                                                                 for x_hat_s, d_x_hat_s, u_hat_s
                                                                 in zip(x_hat[1:], d_x_hat, u_hat[1:])])))
        return self.C1(mu) * (self.C2(mu) * int_adjo + int_norm_res_prim)

    def eta_hat(self, mu, phi_N, Lambdaphi_N, expATVtopx0):
        VtopMtopGMxT = self.VtopMtopGMxT(mu)
        VtopGMxT = self.VtopGMxT(mu)
        return np.sqrt(np.abs(expATVtopx0.T @ self.M15 @ expATVtopx0 + self.xTtopMtopGMxT(mu)
                              + phi_N.T @ self.M12 @ phi_N + Lambdaphi_N.T @ self.M15 @ Lambdaphi_N
                              - 2 * VtopMtopGMxT.T @ expATVtopx0 - 2 * phi_N.T @ self.M13 @ expATVtopx0
                              - 2 * expATVtopx0.T @ self.M15 @ Lambdaphi_N + 2 * VtopGMxT.T @ phi_N
                              + 2 * VtopMtopGMxT.T @ Lambdaphi_N + 2 * phi_N.T @ self.M13 @ Lambdaphi_N))

    def reconstruct_state(self, x):
        return np.array([self.Vpr @ x_s for x_s in x])

    def _solve_reduced_free_dynamics(self, mu):
        A_hat_pr = self.parametrized_A_pr(mu)
        Vtopx0red = self.Vtopx0(mu)
        x_hat_free_dynamics, d_x_hat_free_dynamics = solve_homogeneous_system(Vtopx0red, self.full_model.T,
                                                                              self.full_model.nt, A_hat_pr,
                                                                              return_time_derivative=True,
                                                                              mass_matrix=self.mass_matrix_pr)
        return x_hat_free_dynamics, d_x_hat_free_dynamics

    def estimate_free_dynamics_error(self, mu, return_quantities=False):
        x_hat_free_dynamics, d_x_hat_free_dynamics = self._solve_reduced_free_dynamics(mu)
        dt = self.T / self.nt
        free_dynamics_error = self.delta_prim(mu, x_hat_free_dynamics, d_x_hat_free_dynamics,
                                              np.zeros((self.full_model.nt, self.m)), dt)
        free_dynamics_error *= self.full_model.Mnorm

        if return_quantities:
            return free_dynamics_error, (x_hat_free_dynamics[-1], )
        return free_dynamics_error

    def _compute_reduced_gramian(self, mu, phi_N):
        A_hat_pr = self.parametrized_A_pr(mu)
        A_hat_ad = self.parametrized_A_ad(mu)
        B_hat_pr = self.parametrized_B_pr(mu)
        B_hat_ad = self.parametrized_B_ad(mu)
        M11phi_N = self.M11 @ phi_N
        phi_hat, d_phi_hat = solve_adjoint(M11phi_N, self.full_model.T, self.full_model.nt, A_hat_ad,
                                           return_time_derivative=True, mass_matrix=self.mass_matrix_ad)
        u_hat = get_control_from_adjoint(phi_hat, B_hat_ad, self.R_chol, self.full_model.T)
        x_hat_zero_initial, d_x_hat_zero_initial = solve_system(np.zeros(self.kpr), self.full_model.T,
                                                                self.full_model.nt, A_hat_pr, B_hat_pr, u_hat,
                                                                return_time_derivative=True,
                                                                mass_matrix=self.mass_matrix_pr)
        return x_hat_zero_initial, d_x_hat_zero_initial, u_hat, phi_hat, d_phi_hat

    def compute_reduced_gramian_application(self, mu, phi_N):
        return - self._compute_reduced_gramian(mu, phi_N)[0][-1]

    def estimate_gramian_error(self, mu, phi_N, return_quantities=False):
        x_hat_zero_initial, d_x_hat_zero_initial, u_hat, phi_hat, d_phi_hat = self._compute_reduced_gramian(mu, phi_N)
        dt = self.T / self.nt
        gramian_error = self.delta_lambda(mu, phi_N, x_hat_zero_initial, d_x_hat_zero_initial, u_hat, phi_hat,
                                          d_phi_hat, dt)

        if return_quantities:
            return gramian_error, (-x_hat_zero_initial[-1], )
        return gramian_error

    def estimate_reduced_error(self, mu, phi_N, quantities=None):
        if quantities is not None:
            x_hat_free_dynamics_T = self._solve_reduced_free_dynamics(mu)[0][-1]
            Lambdaphi_N = self.compute_reduced_gramian_application(mu, phi_N)
        else:
            Lambdaphi_N, x_hat_free_dynamics_T = quantities

        reduced_error_estimate = self.eta_hat(mu, phi_N, Lambdaphi_N, x_hat_free_dynamics_T)

        return reduced_error_estimate

    def estimate_error(self, mu, phi_N, return_error_components=False):
        assert phi_N.shape == (self.N, )
        with self.logger.block(f"Estimating error for mu={mu} ...") if self.log_detailed_errors else nullcontext():
            free_dynamics_error, (x_hat_free_dynamics_T, ) = self.estimate_free_dynamics_error(mu,
                                                                                               return_quantities=True)
            free_dynamics_error *= self.full_model.const
            if self.log_detailed_errors:
                self.logger.info(f"Primal error free dynamics: {free_dynamics_error}")
            gramian_error, (Lambdaphi_N, ) = self.estimate_gramian_error(mu, phi_N, return_quantities=True)
            gramian_error *= self.full_model.Mnorm
            gramian_error *= self.full_model.const
            if self.log_detailed_errors:
                self.logger.info(f"Gramian error: {gramian_error}")
            reduced_error_estimator = self.estimate_reduced_error(mu, phi_N,
                                                                  quantities=(Lambdaphi_N, x_hat_free_dynamics_T))
            reduced_error_estimator *= self.full_model.const
            if self.log_detailed_errors:
                self.logger.info(f"Reduced error estimator: {reduced_error_estimator}")
            total_error_estimate = free_dynamics_error + gramian_error + reduced_error_estimator
            if self.log_detailed_errors:
                self.logger.info(f"Total error: {total_error_estimate}")

        if return_error_components:
            return total_error_estimate, {"free_dynamics_error": free_dynamics_error,
                                          "gramian_error": gramian_error,
                                          "reduced_error_estimator": reduced_error_estimator}
        return total_error_estimate

    def solve(self, mu, return_additional_data=False):
        if self.N > 0:
            A_hat_pr = self.parametrized_A_pr(mu)
            A_hat_ad = self.parametrized_A_ad(mu)
            B_hat_pr = self.parametrized_B_pr(mu)
            B_hat_ad = self.parametrized_B_ad(mu)
            Vtopx0red = self.Vtopx0(mu)
            assert Vtopx0red.shape == (self.kpr, )

            expATVtopx0 = solve_homogeneous_system(Vtopx0red, self.full_model.T, self.full_model.nt, A_hat_pr,
                                                   mass_matrix=self.mass_matrix_pr)[-1]

            def specialized_get_state_from_final_time_adjoint(phiT, x0, T, nt, A_pr, A_ad, B_pr, B_ad, R_chol,
                                                              mass_matrix_pr=None, mass_matrix_ad=None):
                """Computes the state trajectory from the final time adjoint."""
                phi = solve_adjoint(phiT, T, nt, A_ad, mass_matrix=mass_matrix_ad)
                u = get_control_from_adjoint(phi, B_ad, R_chol, self.full_model.T)
                return solve_system(x0, T, nt, A_pr, B_pr, u, mass_matrix=mass_matrix_pr)

            temp_mat = np.array([- specialized_get_state_from_final_time_adjoint(phi, np.zeros(self.kpr),
                                                                                 self.full_model.T, self.full_model.nt,
                                                                                 A_hat_pr, A_hat_ad, B_hat_pr, B_hat_ad,
                                                                                 self.R_chol,
                                                                                 mass_matrix_pr=self.mass_matrix_pr,
                                                                                 mass_matrix_ad=self.mass_matrix_ad)[-1]
                                 for phi in self.M11.T]).T

            assert temp_mat.shape == (self.kpr, self.N)

            ATGA = (self.M12
                    + self.M14 @ temp_mat
                    + (self.M14 @ temp_mat).T
                    + temp_mat.T @ self.M15 @ temp_mat)
            assert ATGA.shape == (self.N, self.N)
            ATGb = (self.M14 @ expATVtopx0
                    - self.VtopGMxT(mu)
                    + temp_mat.T @ self.M15 @ expATVtopx0
                    - temp_mat.T @ self.VtopMtopGMxT(mu))
            assert ATGb.shape == (self.N, )

            try:
                phi_red = np.linalg.solve(ATGA, ATGb)
            except np.linalg.LinAlgError as e:
                if "Singular matrix" in str(e):
                    print(f"Error: {e}")
                    print(f"ATGA: {ATGA}")
                    print(f"Norm ATGA: {np.linalg.norm(ATGA)}")
                    print(f"ATGb: {ATGb}")
                    print(f"Norm ATGb: {np.linalg.norm(ATGb)}")
                    phi_red, res, rank, svals = np.linalg.lstsq(ATGA, ATGb, rcond=None)
                    print(f"Least squares solution: {phi_red}")
                    print(f"Residual: {res}")
                    print(f"Rank: {rank}")
                    print(f"Singular values: {svals}")
                    print(f"Norm of residual: {np.linalg.norm(ATGA @ phi_red - ATGb)}")
                    # assert np.isclose(np.linalg.norm(ATGA @ phi_red - ATGb), 0.)
                else:
                    raise
        else:
            phi_red = np.zeros(0)
        if return_additional_data:
            return phi_red, {}
        return phi_red

    def _check_biorthogonality(self):
        kpr = self.Wpr.shape[1]
        assert np.allclose(self.Wpr.T @ self.G @ self.Vpr, np.eye(kpr))
        kad = self.Wad.shape[1]
        assert np.allclose(self.Wad.T @ self.G @ self.Vad, np.eye(kad))

    def compute_control(self, mu, phi_red):
        A_hat_ad = self.parametrized_A_ad(mu)
        B_hat_ad = self.parametrized_B_ad(mu)
        return get_control_from_final_time_adjoint(self.M11 @ phi_red, self.T, self.nt, A_hat_ad,
                                                   B_hat_ad, self.R_chol)

    def compute_state(self, mu, u):
        A_hat_pr = self.parametrized_A_pr(mu)
        B_hat_pr = self.parametrized_B_pr(mu)
        x0 = self.parametrized_x0(mu)
        return solve_system(x0, self.T, self.nt, A_hat_pr, B_hat_pr, u, mass_matrix=self.mass_matrix_pr)

    def extend(self, mu=None, phi=None):
        phi_mu = super().extend(mu=mu, phi=phi)
        # Perform additional reduction of state system
        if self.extend_state_system_basis_on_extend:
            assert mu is not None
            self.extend_state_system_basis(mu, phi_mu, run_precompute=False)
        self.precompute()
        return phi_mu

    def extend_state_system_basis(self, mu, phi=None, run_precompute=True):
        if phi is None:
            phi = self.full_model.solve(mu)

        assert self.reduction_strategy in ["inc_hapod", "dist_hapod"]

        np_vector_space = NumpyVectorSpace(self.n)
        np_vector_array_pr = np_vector_space.from_numpy(self.Vpr.T)
        np_vector_array_pr.scal(self.svals_pr)
        u_mu = self.full_model.compute_control(mu, phi)
        primal_trajectory = self.full_model.compute_state(mu, u_mu)
        np_vector_array_pr.append(np_vector_space.from_numpy(primal_trajectory))
        if self.reduction_strategy == "inc_hapod":
            modes_pr, svals_pr, _ = inc_vectorarray_hapod(self.reduction_strategy_parameters["primal"]["steps"],
                                                          np_vector_array_pr,
                                                          self.reduction_strategy_parameters["primal"]["eps"],
                                                          self.reduction_strategy_parameters["primal"]["omega"],
                                                          product=self.G_product)
        elif self.reduction_strategy == "dist_hapod":
            modes_pr, svals_pr, _ = dist_vectorarray_hapod(self.reduction_strategy_parameters["primal"]["num_slices"],
                                                           np_vector_array_pr,
                                                           self.reduction_strategy_parameters["primal"]["eps"],
                                                           self.reduction_strategy_parameters["primal"]["omega"],
                                                           product=self.G_product)
        self.svals_pr = svals_pr

        self.Vpr = modes_pr.to_numpy().T
        assert self.Vpr.shape[0] == self.n
        self.Wpr = self.Vpr

        np_vector_array_ad = np_vector_space.from_numpy(self.Vad.T)
        np_vector_array_ad.scal(self.svals_ad)
        adjoint_trajectory = self.full_model.compute_adjoint(mu, phi)
        np_vector_array_ad.append(np_vector_space.from_numpy(adjoint_trajectory))

        if self.reduction_strategy == "inc_hapod":
            modes_ad, svals_ad, _ = inc_vectorarray_hapod(self.reduction_strategy_parameters["adjoint"]["steps"],
                                                          np_vector_array_ad,
                                                          self.reduction_strategy_parameters["adjoint"]["eps"],
                                                          self.reduction_strategy_parameters["adjoint"]["omega"],
                                                          product=self.G_product)
        elif self.reduction_strategy == "dist_hapod":
            modes_ad, svals_ad, _ = dist_vectorarray_hapod(self.reduction_strategy_parameters["adjoint"]["num_slices"],
                                                           np_vector_array_ad,
                                                           self.reduction_strategy_parameters["adjoint"]["eps"],
                                                           self.reduction_strategy_parameters["adjoint"]["omega"],
                                                           product=self.G_product)
        self.svals_ad = svals_ad

        self.Vad = modes_ad.to_numpy().T
        assert self.Vad.shape[0] == self.n
        self.Wad = self.Vad

        self._check_biorthogonality()
        if run_precompute:
            self.precompute()
