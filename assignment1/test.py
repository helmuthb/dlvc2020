import numpy as np
import pickle
import cv2
from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset

# Test routines
def main():
    data_TRAINING = PetsDataset("/home/helmuth/dlvc/cifar-10-batches-py", Subset.TRAINING)
    data_VALIDATION = PetsDataset("/home/helmuth/dlvc/cifar-10-batches-py", Subset.VALIDATION)
    data_TEST = PetsDataset("/home/helmuth/dlvc/cifar-10-batches-py", Subset.TEST)
    # check length of datasets
    assert(len(data_TRAINING) == 7959)
    assert(len(data_VALIDATION) == 2041)
    assert(len(data_TEST) == 2000)
    # count cats and dogs
    cat_count = 0
    dog_count = 0
    for s in data_TRAINING:
        if s.label == 0:
            cat_count += 1
        else:
            dog_count += 1
    for s in data_TEST:
        if s.label == 0:
            cat_count += 1
        else:
            dog_count += 1
    for s in data_VALIDATION:
        if s.label == 0:
            cat_count += 1
        else:
            dog_count += 1
    assert(cat_count == 6000)
    assert(dog_count == 6000)
    assert(data_TRAINING[0].data.shape == (32,32,3))
    assert(data_TRAINING[0].data.dtype == np.uint8)
    labels = [data_TRAINING[i].label for i in range(10)]
    assert(labels == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    for i in range(10):
        cv2.imwrite('sample' + str(i) + '.png', data_TRAINING[i].data)

if __name__ == "__main__":
    main()
