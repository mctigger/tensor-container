import pytest
import torch
from pytest import raises
from torch.distributions import Uniform

from src.tensorcontainer.tensor_distribution.uniform import TensorUniform
from tests.compile_utils import run_and_compare_compiled


@pytest.mark.parametrize("low", [0.0, 1.0])
@pytest.mark.parametrize("high", [1.0, 2.0])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_uniform_init(low, high, batch_shape):
    if low >= high:  # Skip invalid combinations
        pytest.skip("low must be less than high for Uniform distribution")
    low_tensor = torch.full(batch_shape, low)
    high_tensor = torch.full(batch_shape, high)
    dist = TensorUniform(low_tensor, high_tensor)
    assert isinstance(dist, TensorUniform)
    assert dist.batch_shape == batch_shape
    assert dist.low.shape == batch_shape
    assert dist.high.shape == batch_shape


@pytest.mark.parametrize("low", [0.0, 1.0])
@pytest.mark.parametrize("high", [1.0, 2.0])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_uniform_api_matching(low, high, batch_shape):
    if low >= high:  # Skip invalid combinations
        pytest.skip("low must be less than high for Uniform distribution")
    low_tensor = torch.full(batch_shape, low)
    high_tensor = torch.full(batch_shape, high)
    td_dist = TensorUniform(low_tensor, high_tensor)
    torch_dist = Uniform(low_tensor, high_tensor)

    # Test properties
    assert torch.allclose(td_dist.mean, torch_dist.mean)
    assert torch.allclose(td_dist.variance, torch_dist.variance)
    assert torch.allclose(td_dist.stddev, torch_dist.stddev)
    assert torch.allclose(td_dist.low, torch_dist.low)
    assert torch.allclose(td_dist.high, torch_dist.high)
    assert td_dist.batch_shape == torch_dist.batch_shape
    assert td_dist.event_shape == torch_dist.event_shape

    # Test methods
    sample_shape = (5,)
    sample = td_dist.sample(torch.Size(sample_shape))
    assert sample.shape == sample_shape + batch_shape
    assert torch.allclose(td_dist.log_prob(sample), torch_dist.log_prob(sample))
    assert torch.allclose(td_dist.entropy(), torch_dist.entropy())


def _test_uniform_compile_fn(dist, sample_shape):
    sample = dist.sample(torch.Size(sample_shape))
    log_prob = dist.log_prob(sample)
    entropy = dist.entropy()
    mean = dist.mean
    variance = dist.variance
    stddev = dist.stddev
    low = dist.low
    high = dist.high
    batch_shape = dist.batch_shape
    event_shape = dist.event_shape
    return (
        sample,
        log_prob,
        entropy,
        mean,
        variance,
        stddev,
        low,
        high,
        batch_shape,
        event_shape,
    )


@pytest.mark.parametrize("low", [0.0, 1.0])
@pytest.mark.parametrize("high", [1.0, 2.0])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_uniform_compile(low, high, batch_shape):
    if low >= high:  # Skip invalid combinations
        pytest.skip("low must be less than high for Uniform distribution")
    low_tensor = torch.full(batch_shape, low)
    high_tensor = torch.full(batch_shape, high)
    dist = TensorUniform(
        low_tensor, high_tensor, validate_args=False
    )  # Pass validate_args=False
    sample_shape = (5,)
    run_and_compare_compiled(_test_uniform_compile_fn, dist, sample_shape)


def test_uniform_validate_args():
    with raises(
        ValueError, match="Uniform is not defined when low>= high"
    ):  # Updated regex
        TensorUniform(torch.tensor([1.0]), torch.tensor([0.0]), validate_args=True)
