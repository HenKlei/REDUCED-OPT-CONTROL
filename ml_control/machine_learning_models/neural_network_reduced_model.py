import numbers
import numpy as np
import torch
import torch.nn as nn

from ml_control.machine_learning_models.basic_reduced_model import BasicReducedMachineLearningModel
from ml_control.machine_learning_models.neural_network_utils import FullyConnectedNN, multiple_restarts_training


class NeuralNetworkReducedModel(BasicReducedMachineLearningModel):
    def __init__(self, reduced_model, training_data, validation_ratio=0.1, scale_inputs=False, scale_outputs=True,
                 zero_padding=True):
        super().__init__(reduced_model, training_data, 'NeuralNetworkReducedModel', zero_padding=zero_padding)

        assert zero_padding, 'Only zero padding supported for neural networks!'

        self.validation_ratio = validation_ratio
        self.scaling_parameters = {'min_inputs': None, 'max_inputs': None,
                                   'min_targets': None, 'max_targets': None}
        self.scale_inputs = scale_inputs
        self.scale_outputs = scale_outputs

        self.neural_network = None

    def _scale_input(self, i):
        """Scales the inputs using the stored scaling parameters."""
        if (self.scaling_parameters.get('min_inputs') is not None
                and self.scaling_parameters.get('max_inputs') is not None):
            return ((torch.DoubleTensor(i) - self.scaling_parameters['min_inputs'])
                    / (self.scaling_parameters['max_inputs'] - self.scaling_parameters['min_inputs']))
        return i

    def _scale_target(self, i):
        """Scales the outputs using the stored scaling parameters."""
        if (self.scaling_parameters.get('min_targets') is not None
                and self.scaling_parameters.get('max_targets') is not None):
            return (torch.DoubleTensor(i) * (self.scaling_parameters['max_targets']
                                             - self.scaling_parameters['min_targets'])
                    + self.scaling_parameters['min_targets'])
        return i

    def _update_scaling_parameters(self, sample):
        """Updates the quantities for scaling of inputs and outputs."""
        if self.scale_inputs:
            if self.scaling_parameters['min_inputs'] is not None:
                self.scaling_parameters['min_inputs'] = torch.min(self.scaling_parameters['min_inputs'], sample[0])
            else:
                self.scaling_parameters['min_inputs'] = sample[0]
            if self.scaling_parameters['max_inputs'] is not None:
                self.scaling_parameters['max_inputs'] = torch.max(self.scaling_parameters['max_inputs'], sample[0])
            else:
                self.scaling_parameters['max_inputs'] = sample[0]

        if self.scale_outputs:
            if self.scaling_parameters['min_targets'] is not None:
                self.scaling_parameters['min_targets'] = torch.min(self.scaling_parameters['min_targets'],
                                                                   sample[1])
            else:
                self.scaling_parameters['min_targets'] = sample[1]
            if self.scaling_parameters['max_targets'] is not None:
                self.scaling_parameters['max_targets'] = torch.max(self.scaling_parameters['max_targets'], sample[1])
            else:
                self.scaling_parameters['max_targets'] = sample[1]

    def train(self, hidden_layers=[50, 50, 50], activation_function=nn.Tanh(),
              training_parameters={'epochs': 1000, 'optimizer': torch.optim.LBFGS, 'learning_rate': 1.}):
        """Trains the neural network surrogate."""
        if isinstance(self.training_data[0][0], numbers.Number):
            dim_parameters = 1
        else:
            dim_parameters = len(self.training_data[0][0])
        training_data_torch = []
        for mu, coeffs in self.training_data:
            sample = (torch.DoubleTensor(np.array(mu).reshape(-1)), torch.DoubleTensor(coeffs))
            self._update_scaling_parameters(sample)
            training_data_torch.append(sample)

        number_validation_snapshots = int(len(training_data_torch) * self.validation_ratio)
        # randomly shuffle training data before splitting into two sets
        np.random.shuffle(training_data_torch)
        # split training data into validation and training set
        validation_data_torch = training_data_torch[0:number_validation_snapshots]
        training_data_torch = training_data_torch[number_validation_snapshots + 1:]
        output_dim = training_data_torch[0][1].shape[0]
        self.logger.info('Setting up neural network ...')
        network = FullyConnectedNN(dim_parameters, hidden_layers, output_dim,
                                   activation_function=activation_function).double()
        self.logger.info('Starting neural network training ...')
        network, losses = multiple_restarts_training(training_data_torch, validation_data_torch, network,
                                                     training_parameters=training_parameters,
                                                     scaling_parameters=self.scaling_parameters)
        self.neural_network = network

    def get_coefficients(self, mu):
        """Computes the reduced coefficients for the given parameter using the neural network surrogate."""
        converted_input = torch.from_numpy(np.array([mu])).double()
        converted_input = self._scale_input(converted_input)
        if self.neural_network:
            phi_reduced_coefficients = self.neural_network(converted_input)
        else:
            phi_reduced_coefficients = torch.zeros(len(self.reduced_model.reduced_basis))
        return self._scale_target(phi_reduced_coefficients).detach().numpy().flatten()
