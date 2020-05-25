import numpy as np
import random

from typing import List, Callable

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for
# what this line means
Op = Callable[[np.ndarray], np.ndarray]


def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op


def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample.astype(dtype)

    return op


def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.ravel(sample)

    return op


def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample + val

    return op


def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample * val

    return op


def mean_sd() -> Op:
    '''
    Calculate sample mean and standard deviation
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.array([np.mean(sample), np.std(sample)])

    return op


def scale(mean: float, sd: float) -> Op:
    '''
    Scale the data to training mean & training std
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return (sample - mean) / sd

    return op


def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.transpose(sample, (2, 0, 1))

    return op


def chw2hwc() -> Op:
    '''
    Flip a 3D array with shape CHW to HWC.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return np.transpose(sample, (1, 2, 0))

    return op


def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        if random.choice([True, False]):
            return np.flip(sample, 0)
        else:
            return sample

    return op


def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the
    top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work
    exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        if pad > 0:
            sample = np.pad(sample, [pad, pad, 0], pad_mode)
        if sz > sample.shape[0] or sz > sample.shape[1]:
            raise ValueError(
                f"Sample too small ({sample.shape}) for cropping size {sz}")

    return op
