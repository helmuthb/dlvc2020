from collections import namedtuple

from dlvc.models.linear import LinearClassifier
from dlvc.test import Accuracy

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

# TODO implement steps 1-2

def train_model(lr: float, momentum: float) -> TrainedModel:
    '''
    Trains a linear classifier with a given learning rate (lr) and momentum.
    Computes the accuracy on the validation set.
    Returns both the trained classifier and accuracy.
    '''

    # TODO implement step 3

    clf = LinearClassifier(...)

    n_epochs = 10
    for i in range(n_epochs):
        for batch in train_batches:
            # train classifier
            pass

    accuracy = Accuracy()
    for batch in val_batches:
        # predict and update accuracy
        pass

    return TrainedModel(clf, accuracy)

# TODO implement steps 4-7

