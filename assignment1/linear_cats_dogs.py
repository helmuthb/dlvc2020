from collections import namedtuple

from dlvc.models.linear import LinearClassifier
from dlvc.test import Accuracy
import dlvc.ops as ops
import numpy as np
import torch
import random
from dlvc.datasets.pets import PetsDataset
from dlvc.batches import BatchGenerator
from dlvc.dataset import Subset

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

# initialize RNG for reproducability
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Step 1: load the data sets (TRAIN, VALIDATION & TEST)
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
# Step 2: Create batch generator for each
BATCH_SIZE = 512
train_batches = BatchGenerator(train_data, BATCH_SIZE, True, op)
val_batches = BatchGenerator(val_data, BATCH_SIZE, True, op)
test_batches = BatchGenerator(test_data, BATCH_SIZE, True, op)

def train_model(lr: float, momentum: float) -> TrainedModel:
    '''
    Trains a linear classifier with a given learning rate (lr) and momentum.
    Computes the accuracy on the validation set.
    Returns both the trained classifier and accuracy.
    '''

    # Step 3: train linear classifier, 10 epochs
    clf = LinearClassifier(3072, train_data.num_classes(), lr, momentum, True)

    n_epochs = 10
    for i in range(n_epochs):
        for batch in train_batches:
            # train classifier
            clf.train(batch.data, batch.label)

    accuracy = Accuracy()
    for batch in val_batches:
        # predict and update accuracy
        prediction = clf.predict(batch.data)
        accuracy.update(prediction, batch.label)

    return TrainedModel(clf, accuracy)

# Step 4: random search for good parameter values
best_model = TrainedModel(None, Accuracy()) # accuracy 0
best_lr = -1
best_momentum = -1
NUM_ATTEMPTS = 1000
# output file: CSV of lr, momentum, accuracy
# this is then used for plotting
f = open('results.csv', 'w')
for i in range(NUM_ATTEMPTS):
    # try random lr and momentum in the range of 0 to 1
    lr = random.random()
    momentum = random.random()
    # train model
    model = train_model(lr, momentum)
    # output hyperparameters & validation accuracy
    f.write(f"{lr},{momentum},{model.accuracy.accuracy()}\n")
    # did we improve the accuracy?
    if model.accuracy > best_model.accuracy:
        best_model = model
        best_lr = lr
        best_momentum = momentum
# close output file
f.close()
print(f"""Validation: {best_model.accuracy}
Parameters: momentum={best_momentum}
            lr={best_lr}""")

# calculate accuracy of the model on the validation set
accuracy = Accuracy()
for batch in test_batches:
    # predict and update accuracy
    prediction = best_model.model.predict(batch.data)
    accuracy.update(prediction, batch.label)

print(f"Test: {accuracy}")
