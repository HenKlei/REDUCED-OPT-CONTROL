import numpy as np
import sklearn.gaussian_process as gp

from ml_control.machine_learning_models.basic_reduced_model import BasicReducedMachineLearningModel


class GaussianProcessRegressionReducedModel(BasicReducedMachineLearningModel):
    def __init__(self, reduced_model, training_data, zero_padding=True):
        super().__init__(reduced_model, training_data, 'GaussianProcessRegressionReducedModel',
                         zero_padding=zero_padding)

        assert zero_padding, 'Only zero padding supported for Gaussian Process Regression!'

    def train(self, kernel=gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(1.0, (1e-3, 1e3)),
              n_restarts_optimizer=10, alpha=0.001, normalize_y=True):
        """Trains the Gaussian process regression surrogate."""
        training_data_x = []
        training_data_y = []
        for mu, coeffs in self.training_data:
            training_data_x.append(mu)
            training_data_y.append(coeffs)
        training_data_x = np.array(training_data_x)
        training_data_y = np.array(training_data_y)
        if training_data_x.ndim == 1:
            training_data_x = training_data_x[..., np.newaxis]

        self.logger.info('Setup Gaussian process regressor ...')
        self.model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, alpha=alpha,
                                                 normalize_y=normalize_y)
        self.logger.info('Fit regressor to data ...')
        self.model.fit(training_data_x, training_data_y)

    def get_coefficients(self, mu):
        """Computes the reduced coefficients for the given parameter using the Gaussian process regression surrogate."""
        converted_input = mu.reshape(1, -1)
        return self.model.predict(converted_input).flatten()
