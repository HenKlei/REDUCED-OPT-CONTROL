import numpy as np

from ml_control.logger import getLogger


class BasicReducedMachineLearningModel:
    def __init__(self, reduced_model, training_data, logger_name, zero_padding=True):
        self.reduced_model = reduced_model
        self.parameter_space = self.reduced_model.parameter_space
        self.training_data = training_data
        self._fix_training_data()

        self.zero_padding = zero_padding

        self.logger = getLogger(logger_name, level='INFO')

    def __getattr__(self, name):
        return self.reduced_model.__getattribute__(name)

    def _fix_training_data(self):
        """Manipulates training data such that all targets have the same length by padding with zeros."""
        N = self.reduced_model.N
        fixed_training_data = []
        for elem in self.training_data:
            tmp = np.zeros(N)
            tmp[:len(elem[1])] = elem[1]
            fixed_training_data.append((elem[0], tmp))
        self.training_data = fixed_training_data

    def train(self):
        """Trains the machine learning surrogate."""
        raise NotImplementedError

    def get_coefficients(self, mu):
        """Computes the reduced coefficients for the given parameter using the machine learning surrogate."""
        raise NotImplementedError

    def extend_model(self):
        if self.zero_padding:
            # Add zero padding to machine learning training data
            for i, (mu, coeffs) in enumerate(self.training_data):
                self.training_data[i] = (mu, np.hstack([coeffs, 0.]))
        else:
            # Train a machine learning surrogate for every component of the coefficients individually
            pass

    def solve(self, mu):
        """Solves the machine learning reduced model for the given parameter."""
        return self.get_coefficients(mu)
