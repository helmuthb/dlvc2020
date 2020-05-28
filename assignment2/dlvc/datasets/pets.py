
import pickle
import numpy as np
from ..dataset import Sample, Subset, ClassificationDataset


class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains
        the Python versionnof the CIFAR-10, i.e. files "data_batch_1",
        "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file
        inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from
            "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images
            from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape 32*32*3, in
        BGR channel order.
        '''

        # set of files to load
        files = []
        if subset == Subset.TRAINING:
            # Load training subset - 1 to 4
            files = [fdir + "/data_batch_" + str(i) for i in range(1, 5)]
        elif subset == Subset.VALIDATION:
            files = [fdir + "/data_batch_5"]
        elif subset == Subset.TEST:
            files = [fdir + "/test_batch"]
        else:
            raise ValueError("Parameter 'subset' has to be either "
                             "'TEST', 'VALIDATION' or 'TRAINING'")
        # I also read in the labels
        try:
            with open(fdir + "/batches.meta", 'rb') as labels_file:
                label_names = pickle.load(labels_file, encoding='bytes')
                cat_label = label_names[b'label_names'].index(b'cat')
                dog_label = label_names[b'label_names'].index(b'dog')
        except IOError:
            raise ValueError("File 'batches.meta' not found")
        # I read in the data in standard Python lists
        # and then convert them into numpy arrays.
        images = []
        labels = []
        for file in files:
            # numpy array from file
            try:
                with open(file, 'rb') as file_handle:
                    file_dict = pickle.load(file_handle, encoding='bytes')
            except IOError:
                raise ValueError("File '" + file + "' not found")
            # loop through samples
            for img in file_dict[b'data']:
                # split values of r, g and b
                r = img[0:1024]
                g = img[1024:2048]
                b = img[2048:3072]
                # reorder in numpy array
                img2 = np.dstack((b, g, r))
                # reshape to 32x32x3 and add to list
                images.append(np.reshape(img2, (32, 32, 3)))
            # add labels
            labels = labels + file_dict[b'labels']
        # reorder into tuples, filter for cat and dogs
        data = []
        for i in range(len(labels)):
            if (labels[i] == cat_label or labels[i] == dog_label):
                new_label = 0 if labels[i] == cat_label else 1
                data.append(Sample(i, images[i], new_label))
        # store results in object variables
        self._data = data
        self._num_classes = 2

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        # return length of data
        return len(self._data)

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''

        # return element #idx
        return self._data[idx]

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        # return number of classes
        return self._num_classes
