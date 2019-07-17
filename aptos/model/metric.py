import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_squared_error

# -- const --

MIN_LABEL = 0
MAX_LABEL = 4
LABELS = list(range(MIN_LABEL, MAX_LABEL + 1))
N_LABELS = len(LABELS)


# -- metrics --

def quadratic_weighted_kappa(output, target):
    return cohen_kappa_score(output, target, labels=LABELS, weights='quadratic')


def conf_matrix(output, target):
    cm = confusion_matrix(target, output, labels=LABELS)
    df = pd.DataFrame(cm)
    return pd.concat([df], keys=['label'], names=['pred'])


def accuracy(output, target):
    return (output == target).mean()


def mse(output, target):
    return mean_squared_error(target, output)


def precision_0(output, target):
    return precision(output, target, 0)


def precision_1(output, target):
    return precision(output, target, 1)


def precision_2(output, target):
    return precision(output, target, 2)


def precision_3(output, target):
    return precision(output, target, 3)


def precision_4(output, target):
    return precision(output, target, 4)


def recall_0(output, target):
    return recall(output, target, 0)


def recall_1(output, target):
    return recall(output, target, 1)


def recall_2(output, target):
    return recall(output, target, 2)


def recall_3(output, target):
    return recall(output, target, 3)


def recall_4(output, target):
    return recall(output, target, 4)


# -- util --


def precision(output, target, c):
    output_c = np.round(output) == c
    target_c = target == c
    return (output_c * target_c).sum() / output_c.sum()


def recall(output, target, c):
    output_c = np.round(output) == c
    target_c = target == c
    return (output_c * target_c).sum() / target_c.sum()
