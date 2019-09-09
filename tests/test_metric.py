"""
Test custom implementation of metric is correct, against this version taken from github.
"""
import numpy as np
import pytest


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating

    https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/quadratic_weighted_kappa.py
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def custom_quadratic_weighted_kappa(rater_a, rater_b, n_values):
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    conf_mat = custom_confusion_matrix(rater_a, rater_b, n_values)
    num_scored_items = rater_a.shape[0]

    # get distance matrix
    dmx = dist_mx(n_values)

    # get numerator matrix
    numerators = dmx * conf_mat

    # get denominator matrix
    expected_counts = np.zeros((n_values, n_values))
    hist_rater_a = np.bincount(rater_a, minlength=n_values)
    hist_rater_b = np.bincount(rater_b, minlength=n_values)
    for j in range(n_values):
        expected_counts[:, j] = hist_rater_a * hist_rater_b[j]
    denominators = dmx * expected_counts

    return 1.0 - (numerators.sum() / denominators.sum()) * num_scored_items


def dist_mx(size):
    mx = np.zeros((size, size))
    values = np.linspace(0, 1, size) ** 2
    for i in range(size - 1):
        mx[i, i:] = values[:size - i]
        mx[i:, i] = values[:size - i]
    return mx


@pytest.mark.parametrize('rater_a, rater_b', [
    (np.random.randint(5, size=5), np.random.randint(5, size=5)),
    (np.random.randint(5, size=10), np.random.randint(5, size=10)),
    (np.random.randint(5, size=20), np.random.randint(5, size=20))
])
def test_quadratic_weighted_kappa(rater_a, rater_b):
    min_ = 0
    max_ = 4
    original = quadratic_weighted_kappa(rater_a, rater_b, min_, max_)
    custom = custom_quadratic_weighted_kappa(rater_a, rater_b, max_ + 1)
    assert original == pytest.approx(custom)


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def custom_confusion_matrix(rater_a, rater_b, n_values):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    conf_mat = np.zeros((n_values, n_values))
    for i, a in enumerate(rater_a):
        conf_mat[a, rater_b[i]] += 1
    return conf_mat


@pytest.mark.parametrize('rater_a, rater_b', [
    (np.random.randint(5, size=5), np.random.randint(5, size=5)),
    (np.random.randint(5, size=10), np.random.randint(5, size=10)),
    (np.random.randint(5, size=20), np.random.randint(5, size=20))
])
def test_confusion_matrix(rater_a, rater_b):
    min_ = 0
    max_ = 4
    original = confusion_matrix(rater_a, rater_b, min_, max_)
    custom = custom_confusion_matrix(rater_a, rater_b, max_ + 1)
    print(len(original), len(original[0]))
    print(custom.shape)
    for i in range(max_ + 1):
        for j in range(max_ + 1):
            print(i, j)
            assert original[i][j] == custom[i, j]


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


@pytest.mark.parametrize('ratings', [
    ([0, 1, 1, 2, 0, 1, 1, 4]),
    ([1, 1, 1, 1, 1, 1, 1, 1]),
    ([3, 1, 1, 2, 3, 1, 1, 4]),
    ([4, 1, 3, 2, 2, 1, 1, 4])
])
def test_histogram(ratings):
    min_ = 0
    max_ = 4
    original = histogram(ratings, min_, max_)
    custom = np.bincount(ratings, minlength=max_ + 1)

    assert len(original) == len(custom)
    for i in range(len(original)):
        assert original[i] == custom[i]
