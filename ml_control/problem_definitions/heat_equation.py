import numpy as np

from ml_control.utilities import Lincomb, ParameterSpace
from ml_control.full_model import FullModel


def create_heat_equation_problem(n=100):
    """Creates all required components of a simple heat equation problem."""
    h = 1. / (n + 1)

    A_tilde = -2. * np.eye(n) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    A_mat = A_tilde / h**2
    parametrized_A = Lincomb([lambda mu: mu], [A_mat])

    B_mat = np.zeros((n, 1))
    B_mat[-1] = 1. / h**2
    parametrized_B = Lincomb([lambda mu: mu], [B_mat])

    parametrized_x0 = Lincomb([lambda _: 1.], [np.sin(np.linspace(h, 1.-h, n) * np.pi)])

    parametrized_xT = Lincomb([lambda _: 1.], [np.linspace(h, 1.-h, n)])

    nt = 10 * n
    T = 0.1

    gamma = 1.
    R = gamma * np.eye(1)
    M = np.eye(n)

    parameter_space = ParameterSpace([1], [2])

    const = 1.
    # TODO: Check C1, C2 and G!
    C1 = lambda _: 1.
    C2 = lambda _: 1.
    G = np.eye(n) * h

    return FullModel(T, nt, n, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R, M,
                     parameter_space, np.eye(n), C1=C1, C2=C2, const=const, G=G, title="Heat equation problem")


def create_time_varying_heat_equation_problem(n=100):
    """Creates all required components of a simple heat equation problem."""
    h = 1. / (n + 1)

    A_tilde = -2. * np.eye(n) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    A_mat = A_tilde / h**2
    parametrized_A = Lincomb([lambda mu, t: mu * (1. + 10. * t)], [A_mat])

    B_mat = np.zeros((n, 1))
    B_mat[-1] = 1. / h**2
    parametrized_B = Lincomb([lambda mu, t: mu * (1. + 10. * t)], [B_mat])

    parametrized_x0 = Lincomb([lambda _: 1.], [np.sin(np.linspace(h, 1.-h, n) * np.pi)])

    parametrized_xT = Lincomb([lambda _: 1.], [np.linspace(h, 1.-h, n)])

    nt = 10 * n
    T = 0.1

    gamma = 1.
    R = gamma * np.eye(1)
    M = np.eye(n)

    parameter_space = ParameterSpace([1], [2])

    # TODO: Check C1, C2 and G!
    C1 = lambda _: 1.
    C2 = lambda _: 1.
    G = np.eye(n) * h

    return FullModel(T, nt, n, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R, M,
                     parameter_space, np.eye(n), C1=C1, C2=C2, G=G, title="Time-varying heat equation problem")


def create_heat_equation_problem_with_two_parameters(n=100):
    """Creates all required components of a heat equation problem with two parameters."""
    h = 1. / (n + 1)

    A_tilde = -2. * np.eye(n) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    parametrized_A = Lincomb([lambda mu: mu[0]], [A_tilde / h**2])

    B_mat = np.zeros((n, 1))
    B_mat[-1] = 1. / h**2
    parametrized_B = Lincomb([lambda mu: mu[0]], [B_mat])

    parametrized_x0 = Lincomb([lambda _: 1.], [np.sin(np.linspace(h, 1.-h, n) * np.pi)])

    parametrized_xT = Lincomb([lambda mu: mu[1]], [np.linspace(h, 1.-h, n)])

    nt = 30 * n
    T = 0.1

    gamma = 0.1
    R = gamma * np.eye(1)
    M = np.eye(n)
    G = np.eye(n) * h

    parameter_space = ParameterSpace([1., 0.5], [2., 1.5])

    return FullModel(T, nt, n, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R, M,
                     parameter_space, np.eye(n), G=G, title="Heat equation problem with two parameters")


def create_heat_equation_problem_complex(n=100):
    """Creates all required components of a heat equation problem with two parameters and two controls."""
    h = 1. / (n + 1)

    A_tilde = -2. * np.eye(n) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    parametrized_A = Lincomb([lambda mu: mu[0]], [A_tilde / h**2])

    B_mat = np.zeros((n, 2))
    B_mat[0, 0] = 1. / h**2
    B_mat[-1, 1] = 1. / h**2
    parametrized_B = Lincomb([lambda mu: mu[0]], [B_mat])

    parametrized_x0 = Lincomb([lambda _: 1.], [np.sin(np.linspace(h, 1.-h, n) * np.pi)])

    parametrized_xT = Lincomb([lambda mu: mu[1]], [np.linspace(h, 1.-h, n)])

    nt = 30 * n
    T = 0.1

    gamma_1 = 0.125
    gamma_2 = 0.25
    R = np.diag([gamma_1, gamma_2])
    M = np.eye(n)
    G = np.eye(n) * h

    parameter_space = ParameterSpace([1., 0.5], [2., 1.5])

    return FullModel(T, nt, n, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R, M,
                     parameter_space, np.eye(n), G=G, title="Complex heat equation problem")
