from collections import namedtuple

from dlvc.models.pytorch import CnnClassifier
from dlvc.test import Accuracy
import dlvc.ops as ops
import numpy as np
import torch
import torch.nn as nn
import random
from dlvc.datasets.pets import PetsDataset
from dlvc.batches import BatchGenerator
from dlvc.dataset import Subset

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

# initialize RNG for reproducability
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Batch size to be used
BATCH_SIZE = 128

# Step 1: load the data sets (TRAIN, VALIDATION)
train_data = PetsDataset("../cifar-10-batches-py", Subset.TRAINING)
val_data = PetsDataset("../cifar-10-batches-py", Subset.VALIDATION)

# Operations to standardize
# First experiment: scale to [-1,1]
op1 = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
    ops.hwc2chw()
])
# Second experiment: scale to sample mean=0, sd=1
# calculate average training sample mean & sd
op_calc = ops.chain([
    ops.type_cast(np.float32),
    ops.mean_sd()
])
# using batch generator (could do it directly but I'm lazy)
train_full_batch_gen = BatchGenerator(
    train_data,
    len(train_data),
    False,
    op_calc)
train_full_batch = next(b for b in train_full_batch_gen)
train_mean_sd = np.mean(train_full_batch.data, axis=0)
# create operation to scale
op2 = ops.chain([
    ops.type_cast(np.float32),
    ops.scale(train_mean_sd[0], train_mean_sd[1]),
    ops.hwc2chw()
])

# Step 2: Create batch generator for each
train_batches = BatchGenerator(train_data, BATCH_SIZE, True, op2)
val_batches = BatchGenerator(val_data, BATCH_SIZE, True, op2)


# Step 3: Define PyTorch CNN
class CatsDogsModel(nn.Module):
    def __init__(self):
        super(CatsDogsModel, self).__init__()
        # Our network (one layer after the other in a list)
        self.layers = nn.Sequential(
            # we start with a 7x7 kernel, padding 3, stride=2
            # creating 64 feature maps
            # (m, 3, 32, 32) -> (m, 64, 16, 16)
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
            # ReLU as activation
            nn.ReLU(inplace=True),
            # and MaxPooling for dimension reduction
            # (m, 64, 16, 16) -> (m, 64, 8, 8)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # now comes a "block" - blowing up the features
            # Step 1: Conv2D
            # (m, 64, 8, 8) -> (m, 128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # another ReLU
            nn.ReLU(inplace=True),
            # Step 2: another Conv2D
            # (m, 128, 8, 8) -> (m, 128, 8, 8)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # another ReLU
            nn.ReLU(inplace=True),
            # Step 3: MaxPooling for dimension reduction
            # (m, 128, 8, 8) -> (m, 128, 4, 4)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # now comes another "block" - blowing up the features
            # Step 1: Conv2D
            # (m, 128, 4, 4) -> (m, 256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # another ReLU
            nn.ReLU(inplace=True),
            # Step 2: another Conv2D
            # (m, 256, 4, 4) -> (m, 256, 4, 4)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # another ReLU
            nn.ReLU(inplace=True),
            # Step 3: MaxPooling for dimension reduction
            # (m, 256, 4, 4) -> (m, 256, 2, 2)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Average pool
            # (m, 256, 2, 2) -> (m, 256, 1, 1)
            nn.AvgPool2d(kernel_size=2),
            # Flatten
            # (m, 256, 1, 1) -> (m, 256)
            nn.Flatten(),
            # Linear layer for 2 classes
            # (m, 256) -> (m, 2)
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # run the steps ...
        return self.layers.forward(x)


def train_model(lr: float, wd: float) -> TrainedModel:
    '''
    Trains a CNN classifier with a given learning rate (lr) and weight decay.
    Computes the accuracy on the validation set.
    Returns both the trained classifier and accuracy.
    '''

    # Step 4: wrap into CnnClassifier
    net = CatsDogsModel()
    # check whether GPU support is available
    if torch.cuda.is_available():
        net.cuda()
    clf = CnnClassifier(net, (0, 3, 32, 32), train_data.num_classes(), lr, wd)

    # Step 5: train in 100 epochs
    n_epochs = 100
    for i in range(n_epochs):
        print(f"epoch {i+1}")
        # reset list of per-epoch training loss
        loss_train = []
        for batch in train_batches:
            # train classifier
            loss_train.append(clf.train(batch.data, batch.label))
        # output as requested
        loss_train = np.array(loss_train)
        print(f" train loss: {np.mean(loss_train):5.3f} +- "
              f"{np.std(loss_train):5.3f}")

        # calculate validation accuracy
        accuracy = Accuracy()
        for batch in val_batches:
            # predict and update accuracy
            prediction = clf.predict(batch.data)
            accuracy.update(prediction, batch.label)
        print(f" val acc: {accuracy}")

    return TrainedModel(clf, accuracy)


model = train_model(0.1, 0.001)
