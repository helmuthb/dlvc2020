import random
import math
import numpy as np
from .dataset import Dataset
from .ops import Op

import typing

class Batch:
    '''
    A (mini)batch generated by the batch generator.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.data = None
        self.label = None
        self.idx = None

class BatchGenerator:
    '''
    Batch generator.
    Returned batches have the following properties:
      data: numpy array holding batch data of shape (s, SHAPE_OF_DATASET_SAMPLES).
      label: numpy array holding batch labels of shape (s, SHAPE_OF_DATASET_LABELS).
      idx: numpy array with shape (s,) encoding the indices of each sample in the original dataset.
    '''

    def __init__(self, dataset: Dataset, num: int, shuffle: bool, op: Op=None):
        '''
        Ctor.
        Dataset is the dataset to iterate over.
        num is the number of samples per batch. the number in the last batch might be smaller than that.
        shuffle controls whether the sample order should be preserved or not.
        op is an operation to apply to input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values, such as if num is > len(dataset).
        '''

        # error detection
        # 1. raise TypeError on invalid argument types
        if not isinstance(dataset, Dataset):
            raise TypeError("Argument `dataset` must be a Dataset")
        if not isinstance(num, int):
            raise TypeError("Argument `num` must be an integer")
        if not isinstance(shuffle, bool):
            raise TypeError("Argument `shuffle` must be a boolean")
        # No check on op, as parameterized generics cannot be
        # used with class or instance checks
        # 2. raise ValueError on invalid argument values
        if num > len(dataset):
            raise ValueError("Number of samples in batch must be smaller than total number")
        # save number of samples
        self._num = num
        # save dataset
        self._dataset = dataset
        # shuffle flag
        self._shuffle = shuffle
        # length
        self._len = len(dataset)
        # operation
        self._op = op

    def __len__(self) -> int:
        '''
        Returns the number of batches generated per iteration.
        '''

        return int(math.ceil(self._len / self._num))

    def __iter__(self) -> typing.Iterable[Batch]:
        '''
        Iterate over the wrapped dataset, returning the data as batches.
        '''

        # initialize
        idx = list(range(self._len))
        if self._shuffle:
            random.shuffle(idx)
        batch_data = []
        batch_label = []
        batch_idx = []
        for cur in idx: 
            # do we have a batch to yield?
            if len(batch_data) >= self._num:
                # yield batch
                batch = Batch()
                batch.data = np.array(batch_data)
                batch.label = np.array(batch_label)
                batch.idx = np.array(batch_idx)
                yield batch
                # reset lists
                batch_data = []
                batch_label = []
                batch_idx = []
            # add next element to lists
            elem = self._dataset[cur]
            if self._op:
                batch_data.append(self._op(elem.data))
            else:
                batch_data.append(elem.data)
            batch_label.append(elem.label)
            batch_idx.append(elem.idx)
        # we are finished with the loop but maybe not all elements
        # were emitted.
        if len(batch_data) > 0:
            # yield last batch
            batch = Batch()
            batch.data = np.array(batch_data)
            batch.label = np.array(batch_label)
            batch.idx = np.array(batch_idx)
            yield batch
