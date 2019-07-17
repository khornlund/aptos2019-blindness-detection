import pytest

import torch
import torch.nn as nn

from aptos.model.loss import WassersteinLoss


@pytest.fixture(scope='session')
def loss():
    return WassersteinLoss(5, reduction='mean', quadratic=True)


@pytest.mark.parametrize('a, b, expected', [
    (
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        0
    ),
    (
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        (0.8 + 0.6 + 0.4 + 0.2)
    ),
    (
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        (0.2 + 0.4 + 0.4 + 0.2)
    ),
    (
        [0.1, 0.1, 0.2, 0.3, 0.3],
        [0.0, 0.0, 0.0, 0.5, 0.5],
        (0.1 + 0.2 + 0.4 + 0.2)
    ),
])
def test_cdf_distance_single_linear(loss, a, b, expected):
    print(a, b, expected)
    ta = torch.Tensor(a)
    tb = torch.Tensor(b)
    assert ta.sum() == 1
    assert tb.sum() == 1
    assert loss.cdf_distance_single_linear(ta, tb).item() == pytest.approx(expected)


@pytest.mark.parametrize('a, b, expected', [
    (
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        0
    ),
    (
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        ( 0.8**2 + 0.6**2 + 0.4**2 + 0.2**2 )
    ),
    (
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        ( 0.2**2 + 0.4**2 + 0.4**2 + 0.2**2 )
    ),
    (
        [0.1, 0.1, 0.3, 0.3, 0.2],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        ( 0.1**2 + 0.2**2 + 0.5**2 + 0.2**2 )
    ),
    (
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        (0.5**2 + 0.5**2)
    ),
])
def test_cdf_distance_single_quadratic(loss, a, b, expected):
    print(a, b, expected)
    ta = torch.Tensor(a)
    tb = torch.Tensor(b)
    assert ta.sum() == 1
    assert tb.sum() == 1
    assert loss.cdf_distance_single_quadratic(ta, tb).item() == pytest.approx(expected)


@pytest.mark.parametrize('a, b, expected', [
    (
        [[0.0, 1.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]],
        [[0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]],
        [0, (0.2 * 1 + 0.2 * 2 + 0.2 * 3 + 0.2 * 4)]
    ),
    (
        [[0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.1, 0.2, 0.3, 0.3]],
        [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 0.5]],
        [(0.2 + 0.4 + 0.4 + 0.2), (0.1 + 0.2 + 0.4 + 0.2)]
    ),
])
def test_cdf_distance_batch_linear(loss, a, b, expected):
    print(a, b, expected)
    ta = torch.Tensor(a)
    tb = torch.Tensor(b)
    result = loss.cdf_distance_batch_linear(ta, tb)
    assert result[0].item() == pytest.approx(expected[0])
    assert result[1].item() == pytest.approx(expected[1])


@pytest.mark.parametrize('a, b, expected', [
    (
        [[0.0, 1.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]],
        [[0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]],
        [0, ( 0.8**2 + 0.6**2 + 0.4**2 + 0.2**2 )]
    ),
    (
        [[0.2, 0.2, 0.2, 0.2, 0.2], [0.1, 0.1, 0.3, 0.3, 0.2]],
        [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]],
        [( 0.2**2 + 0.4**2 + 0.4**2 + 0.2**2 ), ( 0.1**2 + 0.2**2 + 0.5**2 + 0.2**2 )]
    ),
])
def test_cdf_distance_batch_quadratic(loss, a, b, expected):
    print(a, b, expected)
    ta = torch.Tensor(a)
    tb = torch.Tensor(b)
    result = loss.cdf_distance_batch_quadratic(ta, tb)
    print(result[0].item(), result[1].item())
    assert result[0].item() == pytest.approx(expected[0])
    assert result[1].item() == pytest.approx(expected[1])
