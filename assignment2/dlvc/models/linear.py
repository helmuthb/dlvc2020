
from ..model import Model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(Model):
    '''
    Linear classifier without bias.
    Returns softmax class scores (see lecture slides).
    '''

    def __init__(self, input_dim: int, num_classes: int, lr: float,
                 momentum: float, nesterov: bool):
        '''
        Ctor.
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        lr: learning rate to use for training (> 0).
        momentum: momentum to use for training (> 0).
        nesterov: training with or without Nesterov momentum.
        '''

        # running all on CPU is faster here
        self.device = torch.device('cpu')
        # store the input parameters
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        # initialize weight tensor
        self.weights = torch.randn(
            num_classes,
            input_dim,
            requires_grad=True,
            device=self.device)
        # create loss calculator
        self.loss_fn = nn.CrossEntropyLoss()
        # current velocity
        self.velocity = torch.zeros(input_dim, device=self.device)

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple, which is (0, input_dim).
        '''

        # return input shape
        return (0, self.input_dim)

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple,
        which is (num_classes,).
        '''

        # return output shape
        return (self.num_classes,)

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data are the input data, with shape (m, input_dim) and type
        np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and
        num_classes - 1.
        Returns the current cross-entropy loss on the batch.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # Error handling:
        # 1. input must be numpy arrays of specified shape
        if not isinstance(data, np.ndarray):
            raise TypeError('The input `data` must be a numpy array')
        if not isinstance(labels, np.ndarray):
            raise TypeError('The input `labels` must be a numpy array')
        # 2. input must be of the correct shape and compatible with each other
        if len(data.shape) != 2:
            raise ValueError('The input `data` must be a 2-dimensional array')
        if len(labels.shape) != 1:
            raise ValueError('The input `labels` must be a '
                             '2-dimensional array')
        if data.shape[0] != labels.shape[0]:
            raise ValueError('Arrays `labels` and `data` must have the same '
                             'number of rows')
        if data.shape[1] != self.input_dim:
            raise ValueError('The input `data` must be of shape '
                             '(n, num_classes)')
        # 3. data values must be of type np.float32
        if data.dtype != np.float32:
            raise TypeError('The input `data` must be a numpy array of '
                            'float32 values')
        if labels.dtype.kind != 'i':
            raise TypeError('The input `labels` must be a numpy array of '
                            'integer values')
        # all other errors will (hopefully) raise a RuntimeError
        # Nesterov? Then calculate gradient via look-ahead
        if self.nesterov:
            weights = self.weights.add(self.velocity)
        else:
            weights = self.weights
        # Calculate target (prediction)
        prediction = torch.mm(
            torch.from_numpy(data).to(self.device), weights.t())
        # Calculate loss (CrossEntropyLoss includes softmax already)
        loss = self.loss_fn(
            prediction,
            torch.from_numpy(labels).to(self.device))
        self.weights.retain_grad()  # include in the computation graph
        loss.backward()  # compute gradients with backpropagation
        # update velocity
        self.velocity = (self.momentum * self.velocity -
                         self.lr * self.weights.grad)
        # update weights
        self.weights = self.weights.add(self.velocity)
        # return calculated loss
        return loss.item()

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, num_classes) with n being
        the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # Error handling:
        # 1. input must be numpy array of specified shape
        if not isinstance(data, np.ndarray):
            raise TypeError('The input `data` must be a numpy array')
        # 2. input must be of the correct shape
        if len(data.shape) != 2:
            raise ValueError('The input `data` must be a 2-dimensional array')
        if data.shape[1] != self.input_dim:
            raise ValueError('The input `data` must be of shape '
                             '(n, num_classes)')
        # all other errors will (hopefully) raise a RuntimeError
        # Calculate target (prediction) via model
        pred = torch.mm(
            torch.from_numpy(data).to(self.device),
            self.weights.t()).detach()
        return F.softmax(pred, dim=1).cpu().numpy()
