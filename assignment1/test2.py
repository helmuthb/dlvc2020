import numpy as np
import pickle
import cv2
from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
from dlvc.batches import Batch, BatchGenerator
import dlvc.ops as ops

# Test routines
def main():
    data = PetsDataset("/home/helmuth/dlvc/cifar-10-batches-py", Subset.TRAINING)
    # ops chain
    op = ops.chain([
        ops.vectorize(),
        ops.type_cast(np.float32),
        ops.add(-127.5),
        ops.mul(1/127.5),
    ])
    # batch generator #1
    bg1 = BatchGenerator(data, len(data), False)
    assert(len(bg1) == 1)
    # batch generator #2
    bg2 = BatchGenerator(data, 500, False, op)
    assert(len(bg2) == 16)
    # first batch
    cnt = 0
    for batch in bg2:
        cnt += 1
        if cnt < 16:
            assert(batch.data.shape == (500, 3072))
            assert(batch.labels.shape == (500,))
        assert(batch.data.dtype == np.float32)
        assert(np.issubdtype(batch.labels.dtype, np.integer))
        if cnt == 1:
            print("First batch, first sample, not shuffled")
            print(batch.data[0])
    # batch generator #3
    bg3 = BatchGenerator(data, 500, True, op)
    # run 5 times through first sample of shuffled batch generator
    for i in range(5):
        it = iter(bg3)
        print("First batch, first sample, shuffled")
        print(next(it).data[0])

if __name__ == "__main__":
    main()
