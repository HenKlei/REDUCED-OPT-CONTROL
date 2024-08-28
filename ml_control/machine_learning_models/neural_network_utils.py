import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from ml_control.logger import getLogger


class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_function=nn.Tanh()):
        super().__init__()
        self.logger = getLogger('neural_network', level='INFO')

        self.activation_function = activation_function
        layer_sizes = [input_dim] + hidden_layers + [output_dim]

        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(int(layer_sizes[i]), int(layer_sizes[i + 1]))
                            for i in range(len(layer_sizes) - 1)])

        self.activation_function = activation_function

        self.logger.info(f'Architecture of the neural network:\n{self}')

    def forward(self, x):
        """Performs a forward pass through the neural network."""
        for i in range(len(self.layers) - 1):
            x = self.activation_function(self.layers[i](x))
        return self.layers[len(self.layers) - 1](x)


def train_neural_network(training_data, validation_data, neural_network,
                         training_parameters={}, scaling_parameters={}, log_loss_frequency=0):
    """Trains a single neural network using the provided training and validation data."""
    assert isinstance(neural_network, nn.Module)
    assert isinstance(log_loss_frequency, int)

    for data in training_data, validation_data:
        assert isinstance(data, list)
        assert all(isinstance(datum, tuple) and len(datum) == 2 for datum in data)

    def prepare_datum(datum):
        if not (isinstance(datum, torch.DoubleTensor) or isinstance(datum, np.ndarray)):
            return datum.to_numpy()
        return datum

    training_data = [(prepare_datum(datum[0]), prepare_datum(datum[1])) for datum in training_data]
    validation_data = [(prepare_datum(datum[0]), prepare_datum(datum[1])) for datum in validation_data]

    optimizer = optim.LBFGS if 'optimizer' not in training_parameters else training_parameters['optimizer']
    epochs = 1000 if 'epochs' not in training_parameters else training_parameters['epochs']
    assert isinstance(epochs, int) and epochs > 0
    batch_size = 20 if 'batch_size' not in training_parameters else training_parameters['batch_size']
    assert isinstance(batch_size, int) and batch_size > 0
    learning_rate = 1. if 'learning_rate' not in training_parameters else training_parameters['learning_rate']
    assert learning_rate > 0.
    loss_function = (nn.MSELoss() if (training_parameters.get('loss_function') is None)
                     else training_parameters['loss_function'])

    logger = getLogger('neural_network.train_neural_network')

    # LBFGS-optimizer does not support mini-batching, so the batch size needs to be adjusted
    if optimizer == optim.LBFGS:
        batch_size = max(len(training_data), len(validation_data))

    # initialize optimizer, early stopping scheduler and learning rate scheduler
    weight_decay = training_parameters.get('weight_decay', 0.)
    assert weight_decay >= 0.
    optimizer = optimizer(neural_network.parameters(), lr=learning_rate)

    if 'es_scheduler_params' in training_parameters:
        es_scheduler = EarlyStoppingScheduler(len(training_data) + len(validation_data),
                                              **training_parameters['es_scheduler_params'])
    else:
        es_scheduler = EarlyStoppingScheduler(len(training_data) + len(validation_data))
    if training_parameters.get('lr_scheduler'):
        lr_scheduler = training_parameters['lr_scheduler'](optimizer, **training_parameters['lr_scheduler_params'])

    # create the training and validation sets as well as the respective data loaders
    training_dataset = CustomDataset(training_data)
    validation_dataset = CustomDataset(validation_data)
    training_loader = utils.data.DataLoader(training_dataset, batch_size=batch_size)
    validation_loader = utils.data.DataLoader(validation_dataset, batch_size=batch_size)
    dataloaders = {'train':  training_loader, 'val': validation_loader}

    phases = ['train', 'val']

    logger.info('Starting optimization procedure ...')

    if 'min_inputs' in scaling_parameters and 'max_inputs' in scaling_parameters:
        min_inputs = scaling_parameters['min_inputs']
        max_inputs = scaling_parameters['max_inputs']
    else:
        min_inputs = None
        max_inputs = None
    if 'min_targets' in scaling_parameters and 'max_targets' in scaling_parameters:
        min_targets = scaling_parameters['min_targets']
        max_targets = scaling_parameters['max_targets']
    else:
        min_targets = None
        max_targets = None

    # perform optimization procedure
    for epoch in range(epochs):
        losses = {'full': 0.}

        # alternate between training and validation phase
        for phase in phases:
            if phase == 'train':
                neural_network.train()
            else:
                neural_network.eval()

            running_loss = 0.0

            # iterate over batches
            for batch in dataloaders[phase]:
                # scale inputs and outputs if desired
                if min_inputs is not None and max_inputs is not None:
                    diff = max_inputs - min_inputs
                    diff[diff == 0] = 1.
                    inputs = (batch[0] - min_inputs) / diff
                else:
                    inputs = batch[0]
                if min_targets is not None and max_targets is not None:
                    diff = max_targets - min_targets
                    diff[diff == 0] = 1.
                    targets = (batch[1] - min_targets) / diff
                else:
                    targets = batch[1]

                with torch.set_grad_enabled(phase == 'train'):
                    def closure():
                        if torch.is_grad_enabled():
                            optimizer.zero_grad()
                        outputs = neural_network(inputs)
                        loss = loss_function(outputs, targets)
                        if loss.requires_grad:
                            loss.backward()
                        return loss

                    # perform optimization step
                    if phase == 'train':
                        optimizer.step(closure)

                    # compute loss of current batch
                    loss = closure()

                # update overall absolute loss
                running_loss += loss.item() * len(batch[0])

            # compute average loss
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            losses[phase] = epoch_loss

            losses['full'] += running_loss

            if log_loss_frequency > 0 and epoch % log_loss_frequency == 0:
                logger.info(f'Epoch {epoch}: Current {phase} loss of {losses[phase]:.3e}')

            if 'lr_scheduler' in training_parameters and training_parameters['lr_scheduler']:
                lr_scheduler.step()

            # check for early stopping
            if phase == 'val' and es_scheduler(losses, neural_network):
                logger.info(f'Stopping training process early after {epoch + 1} epochs with validation loss '
                            f'of {es_scheduler.best_losses["val"]:.3e} ...')
                return es_scheduler.best_neural_network, es_scheduler.best_losses

    return es_scheduler.best_neural_network, es_scheduler.best_losses


def multiple_restarts_training(training_data, validation_data, neural_network,
                               target_loss=None, max_restarts=10, log_loss_frequency=0,
                               training_parameters={}, scaling_parameters={}):
    """Trains multiple neural networks by restarting the optimization using the provided data."""
    assert isinstance(training_parameters, dict)
    assert isinstance(max_restarts, int) and max_restarts >= 0

    logger = getLogger('neural_network.multiple_restarts_training')

    torch.manual_seed(0)

    # in case no training data is provided, return a neural network
    # that always returns zeros independent of the input
    if len(training_data) == 0 or len(training_data[0]) == 0:
        for layers in neural_network.children():
            for layer in layers:
                torch.nn.init.zeros_(layer.weight)
                layer.bias.data.fill_(0.)
        return neural_network, {'full': None, 'train': None, 'val': None}

    if target_loss:
        logger.info(f'Performing up to {max_restarts} restart{"s" if max_restarts > 1 else ""} '
                    f'to train a neural network with a loss below {target_loss:.3e} ...')
    else:
        logger.info(f'Performing up to {max_restarts} restart{"s" if max_restarts > 1 else ""} '
                    'to find the neural network with the lowest loss ...')

    with logger.block('Training neural network #0 ...'):
        best_neural_network, losses = train_neural_network(training_data, validation_data,
                                                           neural_network, training_parameters,
                                                           scaling_parameters, log_loss_frequency)

    # perform multiple restarts
    for run in range(1, max_restarts + 1):

        if target_loss and losses['full'] <= target_loss:
            logger.info(f'Finished training after {run - 1} restart{"s" if run - 1 != 1 else ""}, '
                        f'found neural network with loss of {losses["full"]:.3e} ...')
            return neural_network, losses

        with logger.block(f'Training neural network #{run} ...'):
            # reset parameters of layers to start training with a new and untrained network
            def reset_parameters_nn(component):
                if hasattr(component, 'children'):
                    for child in component.children():
                        reset_parameters_nn(child)
                try:
                    for child in component:
                        reset_parameters_nn(child)
                except TypeError:
                    pass
                if hasattr(component, 'reset_parameters'):
                    component.reset_parameters()

            reset_parameters_nn(neural_network)

            # perform training
            current_nn, current_losses = train_neural_network(training_data, validation_data,
                                                              neural_network, training_parameters,
                                                              scaling_parameters, log_loss_frequency)

        if current_losses['full'] < losses['full']:
            logger.info(f'Found better neural network (loss of {current_losses["full"]:.3e} '
                        f'instead of {losses["full"]:.3e}) ...')
            best_neural_network = current_nn
            losses = current_losses
        else:
            logger.info(f'Rejecting neural network with loss of {current_losses["full"]:.3e} '
                        f'(instead of {losses["full"]:.3e}) ...')

    if target_loss:
        raise NeuralNetworkTrainingError(f'Could not find neural network with prescribed loss of '
                                          f'{target_loss:.3e} (best one found was {losses["full"]:.3e})!')
    logger.info(f'Found neural network with error of {losses["full"]:.3e} ...')
    return best_neural_network, losses


class EarlyStoppingScheduler:
    def __init__(self, size_training_validation_set, patience=10, delta=0.):
        self.size_training_validation_set = size_training_validation_set
        self.patience = patience
        self.delta = delta

        self.best_losses = None
        self.best_neural_network = None
        self.counter = 0

    def __call__(self, losses, neural_network=None):
        if self.best_losses is None and not np.isnan(losses['full']):
            self.best_losses = losses
            self.best_losses['full'] /= self.size_training_validation_set
            self.best_neural_network = copy.deepcopy(neural_network)
        elif self.best_losses['val'] - self.delta <= losses['val'] or np.isnan(losses['full']):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_losses = losses
            self.best_losses['full'] /= self.size_training_validation_set
            self.best_neural_network = copy.deepcopy(neural_network)
            self.counter = 0

        return False


class CustomDataset(utils.data.Dataset):
    def __init__(self, training_data):
        self.training_data = training_data

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        t = self.training_data[idx]
        return t


class NeuralNetworkTrainingError(Exception):
    """Is raised when training of a neural network fails."""
