import torch
import numpy as np


def distance_matrix(size):
    mx = np.zeros((size, size))
    values = np.linspace(0, 1, size) ** 2
    for i in range(size - 1):
        mx[i, i:] = values[:size - i]
        mx[i:, i] = values[:size - i]
    return mx


MIN_LABEL = 0
MAX_LABEL = 4
N_LABELS = MAX_LABEL - MIN_LABEL + 1
DISTANCE_MATRIX = distance_matrix(N_LABELS)


def quadratic_weighted_kappa(output, target):
    with torch.no_grad():
        preds = output.squeeze(1).clamp(min=MIN_LABEL, max=MAX_LABEL).round()
        return _quadratic_weighted_kappa(
            preds.numpy(),
            target.numpy(),
            N_LABELS)


def _quadratic_weighted_kappa(rater_a, rater_b, n_values):
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, n_values)
    num_scored_items = rater_a.shape[0]

    # get numerator matrix
    numerators = DISTANCE_MATRIX * conf_mat

    # get denominator matrix
    expected_counts = np.zeros((n_values, n_values))
    hist_rater_a = np.bincount(rater_a, minlength=n_values)
    hist_rater_b = np.bincount(rater_b, minlength=n_values)
    for j in range(n_values):
        expected_counts[:, j] = hist_rater_a * hist_rater_b[j]
    denominators = DISTANCE_MATRIX * expected_counts

    return 1.0 - (numerators.sum() / denominators.sum()) * num_scored_items


def confusion_matrix(rater_a, rater_b, n_values):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    conf_mat = np.zeros((n_values, n_values))
    for i, a in enumerate(rater_a):
        conf_mat[a, rater_b[i]] += 1
    return conf_mat
