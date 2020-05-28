import numpy as np
import cv2
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
            sample = np.pad(sample, ((pad, pad), (pad, pad), (0, 0)), pad_mode)
        if sz > sample.shape[0] or sz > sample.shape[1]:
            raise ValueError(
                f"Sample too small ({sample.shape}) for cropping size {sz}")
        # get starting position in both dimensions
        x1 = random.randint(0, sample.shape[0]-sz)
        x1_end = x1 + sz
        x2 = random.randint(0, sample.shape[1]-sz)
        x2_end = x2 + sz
        # return cropped region
        return sample[x1:x1_end, x2:x2_end, :]

    return op


def zoom(sz: int, scale: float) -> Op:
    '''
    With a probability of 66%, zoom into or out of
    an image by a given factor.
    It expects images in HWC with the specified size.
    '''
    mat1 = cv2.getRotationMatrix2D((sz, sz), 0., scale)
    mat2 = cv2.getRotationMatrix2D((sz, sz), 0., 1./scale)

    def op(sample: np.ndarray) -> np.ndarray:
        choice = random.choice([0, 1, 2])
        if choice == 1:
            return cv2.warpAffine(sample, mat1, (sz, sz))
        elif choice == 2:
            return cv2.warpAffine(sample, mat2, (sz, sz))
        else:
            return sample

    return op


def rotate(sz: int, angle: float) -> Op:
    '''
    With a probability of 66%, rotate an image by
    a given angle - left or right.
    It expects images in HWC with the specified size.
    '''
    mat1 = cv2.getRotationMatrix2D((sz, sz), angle, 1)
    mat2 = cv2.getRotationMatrix2D((sz, sz), -angle, 1.)

    def op(sample: np.ndarray) -> np.ndarray:
        choice = random.choice([0, 1, 2])
        if choice == 1:
            return cv2.warpAffine(sample, mat1, (sz, sz))
        elif choice == 2:
            return cv2.warpAffine(sample, mat2, (sz, sz))
        else:
            return sample

    return op


def noise(sigma: float) -> Op:
    '''
    Add random noise, from a normal distribution
    with center 0 and variance sigma^2.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        shape = sample.shape
        noise = np.random.normal(loc=0., scale=sigma, size=shape)
        return sample + noise.reshape(shape).astype(sample.dtype)

    return op


def random_factor(sigma: float) -> Op:
    '''
    Multiply each sample with a random factor,
    drawn from a normal distribution with
    center 1 and variance sigma^2.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        return sample * random.normalvariate(1., sigma)

    return op
