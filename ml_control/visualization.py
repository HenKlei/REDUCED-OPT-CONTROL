import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm


def plot_solution(Z, T, title='Solution state'):
    """Plots a solution as a surface on the space-time domain."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, T, Z.shape[0])
    y = np.linspace(0, 1, Z.shape[1])
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X.T, Y.T, Z, rstride=1, cstride=1,
                           cmap=cm.RdBu, linewidth=0, antialiased=False)
    fig.colorbar(surf)
    fig.suptitle(title)
    plt.show()


def plot_final_time_solutions(states, title='Final time solution state', labels=None, show_plot=True, ax=None):
    """Plots multiple states into a common figure."""
    if not isinstance(states, list):
        states = [states]
    if not labels:
        labels = ['State'] * len(states)
    else:
        assert isinstance(labels, list)
        assert len(labels) == len(states)

    if ax is None:
        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
    x = np.linspace(0, 1, states[0].shape[0])
    ax.grid()
    for c, l in zip(states, labels):
        ax.plot(x, c, label=l)
    if not all(label is None for label in labels):
        try:
            fig.legend()
        except NameError:
            pass
    ax.set_xlabel('x')
    ax.set_ylabel('state')

    if show_plot:
        plt.show()


def plot_controls(u, T, title='Control', labels=None, show_plot=True, ax=None):
    """Plots multiple controls into a common figure."""
    if not isinstance(u, list):
        u = [u]
    if not labels:
        labels = ['Control'] * len(u)
    else:
        assert isinstance(labels, list)
        assert len(labels) == len(u)

    fig_exists = False
    if ax is None:
        fig_exists = True
        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
    x = np.linspace(0, T, u[0].shape[0])
    ax.grid()
    for c, l in zip(u, labels):
        for i in range(c.shape[1]):
            label = l
            if c.shape[1] > 1:
                label += ' component ' + str(i+1)
            ax.plot(x, c[:, i], label=label)
    if not all(label is None for label in labels):
        try:
            fig.legend()
        except NameError:
            pass
    ax.set_xlabel('t')
    ax.set_ylabel('u')
    ax.legend()

    if show_plot:
        plt.show()

    if fig_exists:
        return fig


def plot_final_time_adjoints(phiT, title='Final time adjoint', labels=None, show_plot=True, ax=None):
    """Plots multiple adjoints into a common figure."""
    if not isinstance(phiT, list):
        phiT = [phiT]
    if not labels:
        labels = ['Adjoint'] * len(phiT)
    else:
        assert isinstance(labels, list)
        assert len(labels) == len(phiT)

    if ax is None:
        fig = plt.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
    x = np.linspace(0, 1, phiT[0].shape[0])
    ax.grid()
    for p, l in zip(phiT, labels):
        ax.plot(x, p, label=l)
    if not all(label is None for label in labels):
        try:
            fig.legend()
        except NameError:
            pass
    ax.set_xlabel('x')
    ax.set_ylabel('phi')
    ax.legend()

    if show_plot:
        plt.show()


def animate_solution(x, ylim=(-1, 1), sample_interval=10, title='Animation of solution state'):
    """Animates a time-dependent state."""
    x = x[::sample_interval]
    n = x.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(0, 1), ylim=ylim)
    ax.grid()
    line, = ax.plot(np.linspace(0, 1, n), x[0])

    def animate(i):
        line.set_data(np.linspace(0, 1, n), x[i])
        return ax,

    ani = animation.FuncAnimation(fig, animate, len(x), interval=1000/x.shape[0], blit=True)  # noqa: F841
    fig.suptitle(title)
    plt.show()


def plot_greedy_results(training_parameters, selected_indices, estimated_errors, true_errors, efficiencies, tol,
                        coefficients, labels, reduced_basis, singular_values):
    """Produces multiple plots summarizing the results of the greedy algorithm."""
    if training_parameters.ndim == 1:
        parameter_dim = 1
    else:
        parameter_dim = training_parameters.shape[1]

    assert parameter_dim in [1, 2]

    fig, axs = plt.subplots(3)
    axs[0].semilogy(np.arange(0, len(estimated_errors)), estimated_errors, 'tab:blue', label='Greedy estimated errors')
    axs[0].semilogy(np.arange(0, len(estimated_errors)), true_errors, 'tab:green', label='Greedy true maximum errors')
    axs[0].plot(np.arange(0, len(estimated_errors)), [tol] * len(estimated_errors), 'tab:red', label='Greedy tolerance')
    axs[0].set_xlim((0., len(selected_indices)))
    axs[0].set_xlabel('greedy step')
    axs[0].set_ylabel('maximum estimated error')
    axs[0].set_xticks(np.arange(0, len(selected_indices)) + 1)
    axs[0].legend()
    if parameter_dim == 1:
        axs[1].plot(np.arange(1, len(selected_indices) + 1), training_parameters[selected_indices], 'tab:green',
                    label='Selected parameters')
        axs[1].set_xlim((0., len(selected_indices)))
        axs[1].set_xlabel('greedy step')
        axs[1].set_ylabel('mu')
        axs[1].set_xticks(np.arange(0, len(selected_indices)) + 1)
    else:
        axs[1].remove()
        axs[1] = fig.add_subplot(2, 1, 2, projection='3d')
        axs[1].plot(training_parameters[selected_indices, 0], training_parameters[selected_indices, 1],
                    np.arange(1, len(selected_indices) + 1), label='Selected parameters')
        axs[1].set_zlim((0., len(selected_indices)))
        axs[1].set_zlabel('greedy step')
        axs[1].set_xlabel('mu (component 1)')
        axs[1].set_ylabel('mu (component 2)')
        axs[1].set_zticks(np.arange(0, len(selected_indices)) + 1)
    axs[1].legend()
    axs[2].plot(np.arange(0, len(estimated_errors)), efficiencies, 'tab:blue', label='Maximum efficiencies')
    axs[2].set_xlim((0., len(selected_indices)))
    axs[2].set_xlabel('greedy step')
    axs[2].set_ylabel('maximum efficiency of the error estimator')
    axs[2].set_xticks(np.arange(0, len(selected_indices)) + 1)
    axs[2].legend()
    fig.suptitle('Results of greedy procedure')

    x = np.linspace(0, 1, reduced_basis[0].to_numpy().flatten().shape[0])
    print(parameter_dim)
    if parameter_dim == 1:
        fig2, axs2 = plt.subplots(coefficients[0].shape[1], 2)
    elif parameter_dim == 2:
        fig2, axs2 = plt.subplots(coefficients[0].shape[1])

    for i in range(coefficients[0].shape[1]):
        if parameter_dim == 1:
            axs2elem = axs2[i][0]
        else:
            axs2elem = axs2[i]

        axs2elem.plot(x, reduced_basis[i].to_numpy().flatten())
        axs2elem.set_title(f'(Orthonormalized) Adjoint reduced basis function number {i}')
        axs2elem.set_xlabel('x')
        axs2elem.set_ylabel('phi')
        if parameter_dim == 1:
            if i == 0:
                for coeffs, name in zip(coefficients, labels):
                    axs2[i][1].plot(training_parameters, coeffs[:, i], label=name)
            else:
                for coeffs in coefficients:
                    axs2[i][1].plot(training_parameters, coeffs[:, i])
            axs2[i][1].set_title(f'Coefficients for reduced basis function number {i}')
            axs2[i][1].set_xlabel('parameter mu')
            axs2[i][1].set_ylabel('coefficient')
    fig2.suptitle('Reduced basis for final time adjoint state')
    if parameter_dim == 1:
        fig2.legend()
    plt.show()

    fig = plt.figure()
    fig.suptitle('Singular values of optimal adjoints for training parameters')
    ax = fig.add_subplot(111)
    ax.semilogy(singular_values)
    plt.show()
