import pytest

import torch
import torch.nn as nn


class WassersteinLoss(nn.Module):
    """
    https://stats.stackexchange.com/a/299391
    """
    def __init__(self, n_classes, reduction='mean', quadratic=True):
        self.n = n_classes
        self.reduction = reduction

        if self.reduction is not None and \
           self.reduction != 'mean' and \
           self.reduction != 'sum':
            raise Exception(f'`reduction` must be either `None`, "mean", or "sum"')

        if quadratic:
            self.loss = self.cdf_distance_batch_quadratic
        else:
            self.loss = self.cdf_distance_batch_linear

        super().__init__()

    def forward(self, output, target):
        if self.reduction == 'mean':
            return self.loss(output, target).mean()
        if self.reduction == 'sum':
            return self.loss(output, target).sum()
        return self.loss(output, target)

    # -- batch operations -------------------------------------------------------------------------

    def cdf_distance_batch_linear(self, output, target):
        assert target.shape == output.shape
        cdf_target = torch.cumsum(target, dim=1)
        cdf_estimate = torch.cumsum(output, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        return torch.sum(torch.abs(cdf_diff), dim=1)

    def cdf_distance_batch_quadratic(self, output, target):
        assert target.shape == output.shape
        cdf_target = torch.cumsum(target, dim=1)
        cdf_estimate = torch.cumsum(output, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        return torch.sum(torch.pow(torch.abs(cdf_diff), 2), dim=1)

    # -- 1D operations, used for testing ----------------------------------------------------------

    def cdf_distance_single_linear(self, output, target):
        assert target.shape == output.shape
        cdf_target = torch.cumsum(target, dim=0)
        cdf_estimate = torch.cumsum(output, dim=0)
        cdf_diff = cdf_estimate - cdf_target
        return torch.sum(torch.abs(cdf_diff))

    def cdf_distance_single_quadratic(self, output, target):
        assert target.shape == output.shape
        cdf_target = torch.cumsum(target, dim=0)
        cdf_estimate = torch.cumsum(output, dim=0)
        cdf_diff = cdf_estimate - cdf_target
        return torch.sum(torch.pow(torch.abs(cdf_diff), 2))


@pytest.fixture(scope='session')
def loss():
    return WassersteinLoss(5, reduction='mean')


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
