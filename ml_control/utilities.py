from functools import partial
import inspect
import numpy as np
import scipy.sparse as sps
from scipy.stats import loguniform


class Lincomb:
    def __init__(self, parameter_functions, operators):
        self.parameter_functions = parameter_functions
        self.operators = operators
        self.shape = self.operators[0].shape
        assert len(self.parameter_functions) == len(self.operators)
        self.time_dependent = False
        if len(inspect.signature(self.parameter_functions[0]).parameters) == 2:
            self.time_dependent = True

    def __call__(self, mu, t=None):
        if sps.issparse(self.operators[0]):
            res = sps.csc_matrix(self.shape)
        else:
            res = np.zeros(self.shape)
        if self.time_dependent:
            if t is None:
                return partial(self.__call__, mu=mu)
            assert t is not None
            for theta, op in zip(self.parameter_functions, self.operators):
                res += theta(mu, t) * op
            return res
        for theta, op in zip(self.parameter_functions, self.operators):
            res += theta(mu) * op
        return res


class ParameterSpace:
    def __init__(self, lower_bounds, upper_bounds):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        assert len(self.lower_bounds) == len(self.upper_bounds)
        self.parameter_dimension = len(self.lower_bounds)

    def __len__(self):
        return self.parameter_dimension

    def sample_randomly(self, number_of_samples):
        test_parameters = np.random.rand(number_of_samples, self.parameter_dimension)
        return (test_parameters * np.array([max_ - min_ for (min_, max_) in zip(self.lower_bounds, self.upper_bounds)])
                + np.array([min_ for min_ in self.lower_bounds]))

    def sample_uniformly(self, number_of_samples_per_parameter_dimension):
        x = np.meshgrid(*[np.linspace(min_, max_, number_of_samples_per_parameter_dimension)
                          for (min_, max_) in zip(self.lower_bounds, self.upper_bounds)])
        return np.vstack(list(map(np.ravel, x))).T

    def transform_to_unit_cube(self, parameter):
        return (parameter - np.array(self.lower_bounds)) / (np.array(self.upper_bounds) - np.array(self.lower_bounds))

    def transform_to_parameter_space(self, transformed_parameter):
        return (transformed_parameter * (np.array(self.upper_bounds) - np.array(self.lower_bounds))
                + np.array(self.lower_bounds))


class LogParameterSpace:
    def __init__(self, lower_bound, upper_bound, dim=1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.parameter_dimension = dim

    def __len__(self):
        return self.parameter_dimension

    def sample_randomly(self, number_of_samples):
        return loguniform.rvs(self.lower_bound, self.upper_bound, size=(number_of_samples, self.parameter_dimension))

    def sample_uniformly(self, number_of_samples_per_parameter_dimension):
        x = np.meshgrid(*[np.logspace(np.log10(self.lower_bound), np.log10(self.upper_bound),
                                      number_of_samples_per_parameter_dimension)
                          for _ in range(self.parameter_dimension)])
        return np.vstack(list(map(np.ravel, x))).T

    def transform_to_unit_cube(self, parameter):
        return ((np.log10(parameter) - np.log10(self.lower_bound))
                / (np.log10(self.upper_bound) - np.log10(self.lower_bound)))

    def transform_to_parameter_space(self, transformed_parameter):
        return np.pow(10, (transformed_parameter * (np.log10(self.upper_bound) - np.log10(self.lower_bound))
                           + np.log10(self.lower_bound)))
