import numpy as np
from vkoga.vkoga import VKOGA
from vkoga.kernels import Gaussian

from ml_control.machine_learning_models.basic_reduced_model import BasicReducedMachineLearningModel


class KernelReducedModel(BasicReducedMachineLearningModel):
    def __init__(self, reduced_model, training_data, zero_padding=True, scale_inputs=True):
        super().__init__(reduced_model, training_data, 'KernelReducedModel', zero_padding=zero_padding)
        if scale_inputs:
            def sif(mu):
                return self.parameter_space.transform_to_unit_cube(mu)
        else:
            def sif(mu):
                return mu
        self.scale_inputs_function = sif
        self.outputs_scale = 1.

    def train(self, kernel=Gaussian(1.), greedy_type='p_greedy', tol_p=1e-10, tol_f=1e-10, max_iter=None):
        """Trains the VKOGA surrogate."""
        if self.zero_padding:
            training_data_x = []
            training_data_y = []
            for mu, coeffs in self.training_data:
                training_data_x.append(self.scale_inputs_function(mu))
                training_data_y.append(coeffs)
            training_data_x = np.array(training_data_x)
            training_data_y = self.outputs_scale * np.array(training_data_y)
            if training_data_x.ndim == 1:
                training_data_x = training_data_x[..., np.newaxis]

            self.kernel_model = VKOGA(kernel=kernel, greedy_type=greedy_type, tol_p=tol_p, tol_f=tol_f)
            self.kernel_model = self.kernel_model.fit(training_data_x, training_data_y,
                                                      maxIter=max_iter or len(training_data_x))
        else:
            dim = len(self.reduced_model.reduced_basis)
            training_data_x = [[] for _ in range(dim)]
            training_data_y = [[] for _ in range(dim)]
            for mu, coeffs in self.training_data:
                for i, c in enumerate(coeffs):
                    training_data_x[i].append(self.scale_inputs_function(mu))
                    training_data_y[i].append(c * self.outputs_scale)
            self.kernel_model = []
            for t_x, t_y in zip(training_data_x, training_data_y):
                t_x = np.array(t_x)
                t_y = np.array(t_y)
                if t_x.ndim == 1:
                    t_x = t_x[..., np.newaxis]
                if len(t_x) > 0:
                    k = VKOGA(kernel=kernel, greedy_type=greedy_type, tol_p=tol_p)
                    k = k.fit(t_x, t_y, maxIter=max_iter or len(t_x))
                else:
                    class NullFunction:
                        def predict(self, _):
                            return np.zeros(1)
                    k = NullFunction()
                self.kernel_model.append(k)

    def get_coefficients(self, mu):
        """Computes the reduced coefficients for the given parameter using the VKOGA surrogate."""
        converted_input = self.scale_inputs_function(mu).reshape(1, -1)
        if hasattr(self, "kernel_model"):
            if self.zero_padding:
                return self.kernel_model.predict(converted_input).flatten() / self.outputs_scale
            else:
                return np.array([k.predict(converted_input).flatten()
                                 for k in self.kernel_model]).flatten() / self.outputs_scale
        else:
            return np.zeros(len(self.reduced_model.reduced_basis))
