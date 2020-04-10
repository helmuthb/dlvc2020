
from .model import Model
from .batches import BatchGenerator

import numpy as np

from abc import ABCMeta, abstractmethod

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''

        # set number of samples & correct guesses to 0
        self._cnt_all = 0
        self._cnt_correct = 0

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        # add preduction
        size = len(prediction)
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Both prediction and target must have the same number of rows")
        # get guessed classes
        pred_classes = np.argmax(prediction, axis=1)
        if pred_classes.shape != target.shape:
            raise ValueError("Shape of prediction or target are not supported")
        self._cnt_all += size
        self._cnt_correct += np.sum(pred_classes == target)

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''
        return f"accuracy: {self.accuracy()}"
        # return something like "accuracy: 0.395"

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if not isinstance(other, Accuracy):
            raise TypeError("Can only compare measures of the same type!")
        return self.accuracy() < other.accuracy()

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''

        if not isinstance(other, Accuracy):
            raise TypeError("Can only compare measures of the same type!")
        return self.accuracy() > other.accuracy()

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        # number of correct / number of all
        if self._cnt_all == 0:
            return 0.
        return float(self._cnt_correct) / self._cnt_all