import numpy as np

from ml_control.utilities import Lincomb, ParameterSpace
from ml_control.full_model import FullModel


def create_wave_equation_problem_with_two_controls(n=200, damping_force=0.):
    """Creates all required components of a wave equation problem with two controls."""
    h = 1. / (n // 2. + 1)

    A_tilde = -2. * np.eye(n // 2) + np.diag(np.ones(n // 2 - 1), 1) + np.diag(np.ones(n // 2 - 1), -1)
    parametrized_A = Lincomb([lambda _: 1., lambda mu: mu],
                             [np.block([[np.zeros((n // 2, n // 2)), np.eye(n // 2)],
                                        [np.zeros((n // 2, n // 2)), -damping_force * np.eye(n // 2)]]),
                              np.block([[np.zeros((n // 2, n // 2)), np.zeros((n // 2, n // 2))],
                                        [A_tilde / h**2, np.zeros((n // 2, n // 2))]])])

    B_mat = np.zeros((n, 2))
    B_mat[n // 2, 0] = 1. / h**2
    B_mat[-1, 1] = 1. / h**2
    parametrized_B = Lincomb([lambda mu: mu], [B_mat])

    parametrized_x0 = Lincomb([lambda _: 1.],
                              [np.hstack([np.sin(np.linspace(h, 1.-h, n // 2) * np.pi), np.zeros(n // 2)])])

    parametrized_xT = Lincomb([lambda _: 1.], [np.zeros(n)])

    nt = 10 * n
    T = 3.

    gamma_1 = 0.1
    gamma_2 = 1.
    R = np.diag([gamma_1, gamma_2])
    M = np.eye(n) * 10.
    G = np.eye(n) * h

    parameter_space = ParameterSpace([3], [10])

    return FullModel(T, nt, n, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R, M,
                     parameter_space, np.eye(n), G=G, title="(Damped) Wave equation problem with two controls")


def create_wave_equation_problem(n=200, damping_force=0.):
    """Creates all required components of a damped wave equation problem."""
    h = 1. / (n // 2. + 1)

    A_tilde = -2. * np.eye(n // 2) + np.diag(np.ones(n // 2 - 1), 1) + np.diag(np.ones(n // 2 - 1), -1)
    parametrized_A = Lincomb([lambda _: 1., lambda mu: mu],
                             [np.block([[np.zeros((n // 2, n // 2)), np.eye(n // 2)],
                                        [np.zeros((n // 2, n // 2)), -damping_force * np.eye(n // 2)]]),
                              np.block([[np.zeros((n // 2, n // 2)), np.zeros((n // 2, n // 2))],
                                        [A_tilde / h**2, np.zeros((n // 2, n // 2))]])])

    B_mat = np.zeros((n, 1))
    B_mat[-1] = 1. / h**2
    parametrized_B = Lincomb([lambda mu: mu], [B_mat])

    parametrized_x0 = Lincomb([lambda _: 1.],
                              [np.hstack([np.sin(np.linspace(h, 1.-h, n // 2) * np.pi), np.zeros(n // 2)])])

    parametrized_xT = Lincomb([lambda _: 1.], [np.hstack([np.linspace(h, 1.-h, n // 2), np.zeros(n // 2)])])

    nt = 10 * n
    T = 1.

    gamma = 0.1
    R = gamma * np.eye(1)
    M = np.eye(n) * 10.
    G = np.eye(n) * h

    parameter_space = ParameterSpace([3], [10])

    return FullModel(T, nt, n, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R, M,
                     parameter_space, np.eye(n), G=G, title="(Damped) Wave equation problem")
