from collections import namedtuple
from typing import TextIO

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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        # Our network
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


class CatsDogsModelSimple(nn.Module):
    def __init__(self):
        super(CatsDogsModelSimple, self).__init__()
        # Our network - simplified (no block)
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
            # Average pool
            # (m, 64, 8, 8) -> (m, 64, 1, 1)
            nn.AvgPool2d(kernel_size=8),
            # Flatten
            # (m, 64, 1, 1) -> (m, 64)
            nn.Flatten(),
            # Linear layer for 2 classes
            # (m, 64) -> (m, 2)
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # run the steps ...
        return self.layers.forward(x)


class CatsDogsModelComplex(nn.Module):
    def __init__(self):
        super(CatsDogsModelComplex, self).__init__()
        # Our network - even more layers
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
            # now comes a third "block"
            # Step 1: Conv2D
            # (m, 256, 2, 2) -> (m, 512, 2, 2)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # another ReLU
            nn.ReLU(inplace=True),
            # Step 2: another Conv2D
            # (m, 512, 2, 2) -> (m, 512, 2, 2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # another ReLU
            nn.ReLU(inplace=True),
            # Step 3: MaxPooling for dimension reduction
            # (m, 512, 2, 2) -> (m, 512, 1, 1)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Flatten
            # (m, 512, 1, 1) -> (m, 512)
            nn.Flatten(),
            # Linear layer for 2 classes
            # (m, 512) -> (m, 2)
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # run the steps ...
        return self.layers.forward(x)

# add dropout layer
class CatsDogsModelDropout(nn.Module):
    def __init__(self):
        super(CatsDogsModelDropout, self).__init__()
        # Our network
        self.layers = nn.Sequential(
            # we start with a 7x7 kernel, padding 3, stride=2
            # creating 64 feature maps
            # (m, 3, 32, 32) -> (m, 64, 16, 16)
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
            # ReLU as activation
            nn.ReLU(inplace=True),
            # p – probability of an element to be zeroed, Default: 0.5
            nn.Dropout(),
            # and MaxPooling for dimension reduction
            # (m, 64, 16, 16) -> (m, 64, 8, 8)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # now comes a "block" - blowing up the features
            # Step 1: Conv2D
            # (m, 64, 8, 8) -> (m, 128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # another ReLU
            nn.ReLU(inplace=True),
            # p – probability of an element to be zeroed, Default: 0.5
            nn.Dropout(),
            # Step 2: another Conv2D
            # (m, 128, 8, 8) -> (m, 128, 8, 8)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # another ReLU
            nn.ReLU(inplace=True),
            # p – probability of an element to be zeroed
            nn.Dropout(),
            # Step 3: MaxPooling for dimension reduction
            # (m, 128, 8, 8) -> (m, 128, 4, 4)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # now comes another "block" - blowing up the features
            # Step 1: Conv2D
            # (m, 128, 4, 4) -> (m, 256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # another ReLU
            nn.ReLU(inplace=True),
            # p – probability of an element to be zeroed
            nn.Dropout(),
            # Step 2: another Conv2D
            # (m, 256, 4, 4) -> (m, 256, 4, 4)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # another ReLU
            nn.ReLU(inplace=True),
            # p – probability of an element to be zeroed
            nn.Dropout(),
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
    
    
# add dropout layer before last (dense) layer
class CatsDogsModelDropout2(nn.Module):
    def __init__(self):
        super(CatsDogsModelDropout2, self).__init__()
        # Our network
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
            # p – probability of an element to be zeroed
            nn.Dropout(0.25),
            # Linear layer for 2 classes
            # (m, 256) -> (m, 2)
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # run the steps ...
        return self.layers.forward(x)

# =============================================================================
# # loading the pretrained model
# vgg16 = models.vgg16_bn(pretrained=True)
# # Freeze model weights
# for param in model.parameters():
#     param.requires_grad = False
# # checking if GPU is available
# if torch.cuda.is_available():
#     model = model.cuda()
# # Add on classifier
# model.classifier[6] = Sequential(
#                       Linear(4096, 2))
# for param in model.classifier[6].parameters():
#     param.requires_grad = True
# =============================================================================


# Learning rate to use
lr = 0.01
# weight decay to use
wd = 0.0
# Step 4: wrap into CnnClassifier
net = CatsDogsModel()
# check whether GPU support is available
if torch.cuda.is_available():
    net.cuda()
clf = CnnClassifier(net, (0, 3, 32, 32), train_data.num_classes(), lr, wd)

netSimple = CatsDogsModelSimple()
# check whether GPU support is available
if torch.cuda.is_available():
    netSimple.cuda()
clfSimple = CnnClassifier(
    netSimple,
    (0, 3, 32, 32),
    train_data.num_classes(),
    lr,
    wd
)

netComplex = CatsDogsModelComplex()
# check whether GPU support is available
if torch.cuda.is_available():
    netComplex.cuda()
clfComplex = CnnClassifier(
    netComplex,
    (0, 3, 32, 32),
    train_data.num_classes(),
    lr,
    wd
)

netDropout = CatsDogsModelDropout()
# check whether GPU support is available
if torch.cuda.is_available():
    netDropout.cuda()
clfDropout = CnnClassifier(
    netDropout,
    (0, 3, 32, 32),
    train_data.num_classes(),
    lr,
    wd
)

netDropout2 = CatsDogsModelDropout2()
# check whether GPU support is available
if torch.cuda.is_available():
    netDropout2.cuda()
clfDropout2 = CnnClassifier(
    netDropout2,
    (0, 3, 32, 32),
    train_data.num_classes(),
    lr,
    wd
)


# Step 5: train in 100 epochs
def train_model(clf: CnnClassifier, results_file: TextIO) -> TrainedModel:
    '''
    Trains a CNN classifier.
    Computes the accuracy on the validation set.
    Returns both the trained classifier and accuracy.
    '''

    results_file.write('epoch,train_loss,train_loss_sd,val_accuracy\n')
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
        results_file.write(f"{i},{np.mean(loss_train)},")
        results_file.write(f"{np.std(loss_train)},{accuracy.accuracy()}\n")

    return TrainedModel(clf, accuracy)


with open(f'results_{lr}.csv', 'wt') as results_file:
    model = train_model(clfDropout, results_file)
# =============================================================================
# with open(f'results_simple_{lr}.csv', 'wt') as results_file:
#     modelSimple = train_model(clfSimple, results_file)
# with open(f'results_complex_{lr}.csv', 'wt') as results_file:
#     modelComplex = train_model(clfComplex, results_file)
# =============================================================================

print(f"Model with dropout: {model.accuracy}")
# =============================================================================
# print(f"Simple Model: {modelSimple.accuracy}")
# print(f"Complex Model: {modelComplex.accuracy}")
# =============================================================================
