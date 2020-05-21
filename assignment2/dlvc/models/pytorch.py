
from ..model import Model

import numpy as np
import torch
import torch.nn as nn

class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss (torch.nn.CrossEntropyLoss) and SGD (torch.optim.SGD) are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (SGD with e.g. Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.net = net
        self.optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
        first_parameter = next(net.parameters())
        self.is_cuda = first_parameter.is_cuda

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''

        return self.input_shape

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return (self.num_classses,)

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
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
        if len(data.shape) != 4:
            raise ValueError('The input `data` must be a 4-dimensional tensor')
        if len(labels.shape) != 1:
            raise ValueError('The input `labels` must be a 1-dimensional array')
        if data.shape[0] != labels.shape[0]:
            raise ValueError('Arrays `labels` and `data` must have the same number of rows')
        if data.shape[1:] != self.input_shape[1:]:
            raise ValueError('The input `data` must be of shape {self.input_shape} (except the first dimension)')
        # 3. data values must be of type np.float32
        if data.dtype != np.float32:
            raise TypeError('The input `data` must be a numpy array of float32 values')
        # 4. labels must be integer
        if labels.dtype.kind != 'i':
            raise TypeError('The input `labels` must be a numpy array of integer values')
        # 5. labels must be in the range [0, num_classes-1]
        if np.min(labels) < 0 or np.max(labels) >= self.num_classes:
            raise ValueError(f'The input `labels` must be between 0 and {num_classes-1}')
        # all other errors will (hopefully) raise a RuntimeError

        # set training mode on the network
        self.net.train(True)
        # create PyTorch tensors from the data
        pt_data = torch.from_numpy(data)
        pt_labels = torch.from_numpy(labels)
        # copy to GPU if required
        if self.is_cuda:
            pt_data = pt_data.cuda()
            pt_labels = pt_labels.cuda()
        # reset gradients
        self.optimizer.zero_grad()
        outputs = self.net(pt_data)
        loss = self.loss_fn(outputs, pt_labels)
        loss.backward()
        self.optimizer.step()
        # training loss: in loss
        return loss.item()

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        # Error handling:
        # 1. input must be numpy array of specified shape
        if not isinstance(data, np.ndarray):
            raise TypeError('The input `data` must be a numpy array')
        # 2. input must be of the correct shape
        if len(data.shape) != 4:
            raise ValueError('The input `data` must be a 4-dimensional tensor')
        if data.shape[1:] != self.input_shape[1:]:
            raise ValueError('The input `data` must be of shape {self.input_shape} (except the first dimension)')
        # all other errors will (hopefully) raise a RuntimeError
        # Calculate target (prediction) via model
        self.net.eval()
        # create PyTorch tensors from the data
        pt_data = torch.from_numpy(data)
        # copy to GPU if required
        if self.is_cuda:
            pt_data = pt_data.cuda()
        pred = self.net(pt_data).detach()
        # always calling cpu() (it does not cost more than an if would)
        return self.softmax(pred).cpu().numpy()
