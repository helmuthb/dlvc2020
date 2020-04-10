from collections import namedtuple

from dlvc.models.linear import LinearClassifier
from dlvc.test import Accuracy
import dlvc.ops as ops
import numpy as np
from dlvc.datasets.pets import PetsDataset
from dlvc.batches import BatchGenerator
from dlvc.dataset import Subset

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

# load the data sets (TRAIN, VALIDATION & TEST)
train_data = PetsDataset("../cifar-10-batches-py", Subset.TRAINING)
val_data = PetsDataset("../cifar-10-batches-py", Subset.VALIDATION)
test_data = PetsDataset("../cifar-10-batches-py", Subset.TEST)

# Operations to standardize
op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])
# Create batch generator for each
train_batches = BatchGenerator(train_data, 32, True, op)
val_batches = BatchGenerator(val_data, 32, True, op)
test_batches = BatchGenerator(test_data, 32, True, op)

# TODO implement steps 1-2

def train_model(lr: float, momentum: float) -> TrainedModel:
    '''
    Trains a linear classifier with a given learning rate (lr) and momentum.
    Computes the accuracy on the validation set.
    Returns both the trained classifier and accuracy.
    '''

    # TODO implement step 3
    clf = LinearClassifier(3072, train_data.num_classes(), lr, momentum, True)

    n_epochs = 10
    for i in range(n_epochs):
        for batch in train_batches:
            # reshape batch (make each sample flat)
            data = batch.data.reshape(-1, 3072)
            # train classifier
            clf.train(data, batch.label)

    accuracy = Accuracy()
    for batch in val_batches:
        # reshape batch (make each sample flat)
        data = batch.data.reshape(-1, 3072)
        # predict and update accuracy
        prediction = clf.predict(data)
        accuracy.update(prediction, batch.label)

    return TrainedModel(clf, accuracy)

# TODO implement steps 4-7
model = train_model(0.1, 0.9)
print(model.accuracy)
