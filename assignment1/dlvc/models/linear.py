
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

    def __init__(self, input_dim: int, num_classes: int, lr: float, momentum: float, nesterov: bool):
        '''
        Ctor.
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        lr: learning rate to use for training (> 0).
        momentum: momentum to use for training (> 0).
        nesterov: training with or without Nesterov momentum.
        '''

        # store the input parameters
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        # initialize weight tensor
        self.weights = torch.randn(num_classes, input_dim, requires_grad=True)
        # create loss calculator
        self.loss_fn = nn.CrossEntropyLoss()
        # current velocity
        self.velocity = torch.zeros(input_dim)


    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple, which is (0, input_dim).
        '''

        # return input shape
        return (0, self.input_dim)

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        # return output shape
        return (self.num_classes,)

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the current cross-entropy loss on the batch.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # Nesterov? Then calculate gradient via look-ahead
        if self.nesterov:
            weights = self.weights.add(self.velocity)
        else:
            weights = self.weights
        # Calculate target (prediction)
        prediction = torch.mm(torch.from_numpy(data), weights.t())
        # Calculate loss
        loss = self.loss_fn(prediction, torch.from_numpy(labels))
        self.weights.retain_grad() # include this tensor in the computation graph
        loss.backward() # compute gradients with backpropagation
        # update velocity
        self.velocity = self.momentum * self.velocity - self.lr * self.weights.grad
        # update weights
        self.weights = self.weights.add(self.velocity)
        # return calculated loss
        return loss.item()

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # Calculate target (prediction)
        return torch.mm(torch.from_numpy(data), self.weights.t()).detach().numpy()