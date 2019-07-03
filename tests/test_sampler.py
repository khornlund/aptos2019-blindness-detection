import pytest

import numpy as np

from aptos.data_loader import SamplerFactory, CircularList


@pytest.fixture
def factory():
    return SamplerFactory()


@pytest.mark.parametrize('weight_a, weight_b, alpha, expected', [
    (
        [0.1, 0.8, 0.1],
        [1 / 3, 1 / 3, 1 / 3],
        1,
        [0.1, 0.8, 0.1]
    ),
    (
        [0.1, 0.8, 0.1],
        [1 / 3, 1 / 3, 1 / 3],
        0,
        [1 / 3, 1 / 3, 1 / 3],
    ),
    (
        [0.1, 0.8, 0.1],
        [1 / 3, 1 / 3, 1 / 3],
        0.5,
        [13 / 60, 34 / 60, 13 / 60],
    ),
])
def test_samplerfactory_balance_weights(factory, weight_a, weight_b, alpha, expected):
    out = factory.balance_weights(
        np.asarray(weight_a),
        np.asarray(weight_b),
        alpha)
    assert out.shape[0] == len(expected)
    for idx, item in enumerate(out):
        assert item == pytest.approx(expected[idx])


@pytest.mark.parametrize('weights, class_sizes, batch_size, exp_counts, exp_batches', [
    (
        # 10 of each class per batch, expected 12 batches
        [1 / 3, 1 / 3, 1 / 3],
        [60, 90, 120],
        30,
        [10, 10, 10],
        12
    ),
    (
        # 5/5/10 of each class per batch, expected 18 batches
        [1 / 4, 1 / 4, 1 / 2],
        [60, 90, 120],
        20,
        [5, 5, 10],
        18
    ),
    (
        # 5/5/11 of each class per batch, expected 18 batches
        [1 / 4, 1 / 4, 1 / 2],
        [60, 90, 120],
        21,
        [5, 5, 11],
        18
    ),
    (
        # 5/5/11 of each class per batch, expected 20 batches
        [1 / 4, 1 / 4, 1 / 2],
        [60, 90, 220],
        21,
        [5, 5, 11],
        20
    ),
])
def test_samplerfactory_batch_statistics(factory, weights, class_sizes, batch_size, exp_counts,
                                         exp_batches):
    counts, n_batches = factory.batch_statistics(
        np.asarray(weights),
        np.asarray(class_sizes),
        batch_size)

    np.testing.assert_allclose(counts, exp_counts)
    np.testing.assert_allclose(n_batches, exp_batches)


@pytest.mark.parametrize('items, start_idx, end_idx, expected_items', [
    (
        [1],
        0, 5,
        [1, 1, 1, 1, 1]
    ),
    (
        [1, 2, 3],
        0, 6,
        [1, 2, 3, 1, 2, 3]
    ),
    (
        [1, 2],
        0, 5,
        [1, 2, 1, 2, 1]
    )
])
def test_circularlist(items, start_idx, end_idx, expected_items):
    cl = CircularList(items)
    assert cl[start_idx:end_idx] == expected_items
