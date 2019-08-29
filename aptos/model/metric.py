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
    return cohen_kappa_score(round_clip(output), target, labels=LABELS, weights='quadratic')


def conf_matrix(output, target):
    cm = confusion_matrix(target, round_clip(output), labels=LABELS)
    df = pd.DataFrame(cm)
    return pd.concat([df], keys=['label'], names=['pred'])


def accuracy(output, target):
    return (round_clip(output) == target).mean()


def mse(output, target):
    return mean_squared_error(target, output)


# -- util --

def round_clip(output):
    return np.clip(output.round(), a_min=MIN_LABEL, a_max=MAX_LABEL)
