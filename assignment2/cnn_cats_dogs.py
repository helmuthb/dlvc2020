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

# Step 1: load the data sets (TRAIN, VALIDATION)
train_data = PetsDataset("../cifar-10-batches-py", Subset.TRAINING)
val_data = PetsDataset("../cifar-10-batches-py", Subset.VALIDATION)

# Operations to standardize
# TODO: perform experiments, e.g. scaling according to test data
op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
    ops.hwc2chw()
])
# Step 2: Create batch generator for each
BATCH_SIZE = 512
train_batches = BatchGenerator(train_data, BATCH_SIZE, True, op)
val_batches = BatchGenerator(val_data, BATCH_SIZE, True, op)

# Step 3: Define PyTorch CNN
class CatsDogsModel(nn.Module):
    def __init__(self):
        super(CatsDogsModel, self).__init__()
        # we start with a 7x7 kernel, padding 3, stride=2
        # creating 64 feature maps
        # (m, 3, 32, 32) -> (m, 64, 16, 16)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
        # ReLU as activation
        self.relu1 = nn.ReLU(inplace=True)
        # and MaxPooling for dimension reduction
        # (m, 64, 16, 16) -> (m, 64, 8, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # now comes a "block" - blowing up the features
        # Step 1: Conv2D
        # (m, 64, 8, 8) -> (m, 128, 8, 8)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # another ReLU
        self.relu2 = nn.ReLU(inplace=True)
        # Step 2: another Conv2D
        # (m, 128, 8, 8) -> (m, 128, 8, 8)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # another ReLU
        self.relu3 = nn.ReLU(inplace=True)
        # Step 3: MaxPooling for dimension reduction
        # (m, 128, 8, 8) -> (m, 128, 4, 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # now comes another "block" - blowing up the features
        # Step 1: Conv2D
        # (m, 128, 4, 4) -> (m, 256, 4, 4)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # another ReLU
        self.relu4 = nn.ReLU(inplace=True)
        # Step 2: another Conv2D
        # (m, 256, 4, 4) -> (m, 256, 4, 4)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # another ReLU
        self.relu5 = nn.ReLU(inplace=True)
        # Step 3: MaxPooling for dimension reduction
        # (m, 256, 4, 4) -> (m, 256, 2, 2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Average pool
        # (m, 256, 2, 2) -> (m, 256, 1, 1)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        # Flatten
        # (m, 256, 1, 1) -> (m, 256)
        self.flatten = nn.Flatten()
        # Linear layer for 2 classes
        # (m, 256) -> (m, 2)
        self.linear = nn.Linear(256, 2)

    def forward(self, x):
        # run the steps ...
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

def train_model(lr: float, wd: float) -> TrainedModel:
    '''
    Trains a CNN classifier with a given learning rate (lr) and weight decay.
    Computes the accuracy on the validation set.
    Returns both the trained classifier and accuracy.
    '''

    # Step 3: train CNN classifier, 100 epochs
    net = CatsDogsModel()
    net.cuda()
    clf = CnnClassifier(net, (0, 3, 32, 32), train_data.num_classes(), lr, wd)

    n_epochs = 100
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
best_wd = -1
NUM_ATTEMPTS = 1000
# output file: CSV of lr, wd, accuracy
# this is then used for plotting
f = open('results.csv', 'w')
for i in range(NUM_ATTEMPTS):
    # try random lr in the range of 0 to 1 and wd in the range 0 to 0.1
    lr = random.random()
    wd = random.random() / 10.
    # train model
    model = train_model(lr, wd)
    # output hyperparameters & validation accuracy
    f.write(f"{lr},{wd},{model.accuracy.accuracy()}\n")
    # did we improve the accuracy?
    if model.accuracy > best_model.accuracy:
        best_model = model
        best_lr = lr
        best_wd = wd
# close output file
f.close()
print(f"""Validation: {best_model.accuracy}
Parameters: wd={best_wd}
            lr={best_lr}""")
