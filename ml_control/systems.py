import numpy as np
from scipy.linalg import lu_factor, lu_solve, cho_solve
import scipy.sparse as sps

from ml_control.logger import getLogger


def solve_adjoint(phiT, T, nt, A, return_time_derivative=False, mass_matrix=None):
    """Solves the adjoint equation backward in time."""
    n = phiT.shape[0]
    p = np.zeros((nt + 1, n))
    p[nt] = phiT
    dt = T / nt
    inv = None
    if callable(A):
        # Time-variant system!
        if sps.issparse(A(t=0.)):
            solve = sps.linalg.spsolve
            if mass_matrix is None:
                mass_matrix = sps.eye(n)
        else:
            solve = np.linalg.solve
            if mass_matrix is None:
                mass_matrix = np.eye(n)
        for k in range(nt - 1, -1, -1):
            p[k] = solve(mass_matrix.T - dt * A(t=k*dt).T, mass_matrix.T @ p[k + 1])
        if return_time_derivative:
            return p, compute_time_derivative_system(p, T, nt)
        return p

    if sps.issparse(A):
        if mass_matrix is None:
            mass_matrix = sps.eye(n)
        R = mass_matrix.T - dt * A.T
        inv = sps.linalg.splu(R.tocsc())
    else:
        if mass_matrix is None:
            mass_matrix = np.eye(n)
        R = mass_matrix.T - dt * A.T
        lu, piv = lu_factor(R)
    S = mass_matrix.T
    if inv is not None:
        for k in range(nt - 1, -1, -1):
            p[k] = inv.solve(S @ p[k + 1])
    else:
        for k in range(nt - 1, -1, -1):
            p[k] = lu_solve((lu, piv), S @ p[k + 1])
    if return_time_derivative:
        return p, compute_time_derivative_system(p, T, nt)
    return p


def solve_system(x0, T, nt, A, B, u, return_time_derivative=False, mass_matrix=None):
    """Solves the primal system forward in time."""
    n = x0.shape[0]
    x = np.zeros((nt + 1, n))
    x[0] = x0
    dt = T / nt
    inv = None
    if callable(A):
        # Time-variant system!
        if sps.issparse(A(t=0.)):
            solve = sps.linalg.spsolve
            if mass_matrix is None:
                mass_matrix = sps.eye(n)
        else:
            solve = np.linalg.solve
            if mass_matrix is None:
                mass_matrix = np.eye(n)
        for k in range(nt):
            x[k + 1] = solve(mass_matrix - dt * A(t=(k+1)*dt),
                             mass_matrix @ x[k] + dt * B(t=(k+1)*dt) @ u[k + 1])
        if return_time_derivative:
            return x, compute_time_derivative_system(x, T, nt)
        return x

    if sps.issparse(A):
        if mass_matrix is None:
            mass_matrix = sps.eye(n)
        R = mass_matrix - dt * A
        inv = sps.linalg.splu(R.tocsc())
    else:
        if mass_matrix is None:
            mass_matrix = np.eye(n)
        R = mass_matrix - dt * A
        lu, piv = lu_factor(R)
    S = mass_matrix
    if inv is not None:
        for k in range(nt):
            x[k + 1] = inv.solve(S @ x[k] + dt * B @ u[k + 1])
    else:
        for k in range(nt):
            x[k + 1] = lu_solve((lu, piv), S @ x[k] + dt * B @ u[k + 1])
    if return_time_derivative:
        return x, compute_time_derivative_system(x, T, nt)
    return x


def solve_homogeneous_system(x0, T, nt, A, return_time_derivative=False, mass_matrix=None):
    """Solves the homogeneous primal system, i.e. the system without control, forward in time."""
    n = x0.shape[0]
    x = np.zeros((nt + 1, n))
    x[0] = x0
    dt = T / nt
    inv = None
    if callable(A):
        # Time-variant system!
        if sps.issparse(A(t=0.)):
            solve = sps.linalg.spsolve
            if mass_matrix is None:
                mass_matrix = sps.eye(n)
        else:
            solve = np.linalg.solve
            if mass_matrix is None:
                mass_matrix = np.eye(n)
        for k in range(nt):
            x[k + 1] = solve(mass_matrix - dt * A(t=(k+1)*dt),
                             mass_matrix @ x[k])
        if return_time_derivative:
            return x, compute_time_derivative_system(x, T, nt)
        return x

    if sps.issparse(A):
        if mass_matrix is None:
            mass_matrix = sps.eye(n)
        R = mass_matrix - dt * A
        inv = sps.linalg.splu(R.tocsc())
    else:
        if mass_matrix is None:
            mass_matrix = np.eye(n)
        R = mass_matrix - dt * A
        lu, piv = lu_factor(R)
    S = mass_matrix
    if inv is not None:
        for k in range(nt):
            x[k + 1] = inv.solve(S @ x[k])
    else:
        for k in range(nt):
            x[k + 1] = lu_solve((lu, piv), S @ x[k])
    if return_time_derivative:
        return x, compute_time_derivative_system(x, T, nt)
    return x


def compute_time_derivative_system(x, T, nt):
    return (x[1:] - x[:-1]) / (T / nt)


def get_control_from_adjoint(phi, B, R_chol, T):
    """Computes the control from the trajectory of the adjoint."""
    nt = len(phi) - 1
    dt = T / nt
    if callable(B):
        return np.array([-cho_solve(R_chol, B(t=k*dt).T @ p) for k, p in enumerate(phi)])
    return np.array([-cho_solve(R_chol, B.T @ p) for p in phi])


def get_state_from_adjoint(phi, x0, T, nt, A, B, R_chol, mass_matrix=None):
    """Computes the state trajectory from the trajectory of the adjoint."""
    u = get_control_from_adjoint(phi, B, R_chol, T)
    return solve_system(x0, T, nt, A, B, u, mass_matrix=mass_matrix)


def get_state_from_final_time_adjoint(phiT, x0, T, nt, A, B, R_chol, mass_matrix=None):
    """Computes the state trajectory from the final time adjoint."""
    phi = solve_adjoint(phiT, T, nt, A, mass_matrix=mass_matrix)
    return get_state_from_adjoint(phi, x0, T, nt, A, B, R_chol, mass_matrix=mass_matrix)


def get_control_from_final_time_adjoint(phiT, T, nt, A, B, R_chol, mass_matrix=None):
    """Computes the control from the final time adjoint."""
    phi = solve_adjoint(phiT, T, nt, A, mass_matrix=mass_matrix)
    return get_control_from_adjoint(phi, B, R_chol, T)


def solve_optimal_control_problem(x0, xT, T, nt, A, B, R_chol, M, phiT_init, mass_matrix=None,
                                  solver_options={"maxiter": None, "atol": 0.0, "rtol": 1e-8},
                                  solve_exact_controllability_problem=False):
    """Solves the optimal control problem exactly by applying the BICGSTAB algorithm."""
    logger = getLogger("bicgstab", level="INFO")

    x0_T = solve_homogeneous_system(x0, T, nt, A, mass_matrix=mass_matrix)[-1]
    n = x0_T.shape[0]
    if solve_exact_controllability_problem:
        rhs = x0_T - xT
    else:
        rhs = M @ (x0_T - xT)

    def apply_matrix(phiT):
        xT_phiT = get_state_from_final_time_adjoint(phiT, np.zeros_like(phiT), T, nt,
                                                    A, B, R_chol, mass_matrix=mass_matrix)[-1]
        if solve_exact_controllability_problem:
            return - xT_phiT
        else:
            return mass_matrix.T @ phiT - M @ xT_phiT

    def r_apply_matrix(phiT):
        if solve_exact_controllability_problem:
            xT_phiT = get_state_from_final_time_adjoint(phiT, np.zeros_like(phiT), T, nt,
                                                        A, B, R_chol, mass_matrix=mass_matrix)[-1]
            return - xT_phiT
        else:
            xT_phiT = get_state_from_final_time_adjoint(M @ phiT, np.zeros_like(phiT), T, nt,
                                                        A, B, R_chol, mass_matrix=mass_matrix)[-1]
            return mass_matrix @ phiT - xT_phiT

    logger.info("Running BICGSTAB to solve the optimal control problem ...")
    logger.info(f"BICGSTAB solver options: {solver_options}")
    op = sps.linalg.LinearOperator(shape=(n, n), matvec=apply_matrix, rmatvec=r_apply_matrix)
    phiT, info = sps.linalg.bicgstab(op, rhs, **solver_options)

    return phiT
