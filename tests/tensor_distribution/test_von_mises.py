import pytest
import torch
from torch.distributions import VonMises as TorchVonMises

from tensorcontainer.tensor_distribution.von_mises import TensorVonMises


def compile_module(module):
    return torch.compile(module, fullgraph=True)


def _check_sample_log_prob(dist, torch_dist):
    sample = dist.sample()
    assert sample.shape == dist.batch_shape + dist.event_shape
    assert torch.allclose(dist.log_prob(sample), torch_dist.log_prob(sample))


def _check_properties(dist, torch_dist):
    assert dist.batch_shape == torch_dist.batch_shape
    assert dist.event_shape == torch_dist.event_shape
    assert torch.allclose(dist.mean, torch_dist.mean)
    assert torch.allclose(dist.variance, torch_dist.variance)
    # VonMises does not implement entropy
    # assert torch.allclose(dist.entropy(), torch_dist.entropy())
    assert dist.has_rsample == torch_dist.has_rsample
    assert dist.has_enumerate_support == torch_dist.has_enumerate_support
    assert dist.support == torch_dist.support


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("loc", [0.0, 1.0])
@pytest.mark.parametrize("concentration", [0.1, 1.0, 10.0])
def test_von_mises_init(batch_shape, loc, concentration):
    loc_tensor = torch.full(batch_shape, loc)
    concentration_tensor = torch.full(batch_shape, concentration)

    dist = TensorVonMises(loc_tensor, concentration_tensor)
    torch_dist = TorchVonMises(loc_tensor, concentration_tensor)

    assert isinstance(dist, TensorVonMises)
    assert isinstance(dist.dist(), TorchVonMises)
    assert dist.batch_shape == batch_shape
    assert dist.event_shape == torch_dist.event_shape
    assert dist.loc.shape == loc_tensor.shape
    assert dist.concentration.shape == concentration_tensor.shape


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("loc", [0.0, 1.0])
@pytest.mark.parametrize("concentration", [0.1, 1.0, 10.0])
def test_von_mises_properties(batch_shape, loc, concentration):
    loc_tensor = torch.full(batch_shape, loc)
    concentration_tensor = torch.full(batch_shape, concentration)

    dist = TensorVonMises(loc_tensor, concentration_tensor)
    torch_dist = TorchVonMises(loc_tensor, concentration_tensor)

    _check_properties(dist, torch_dist)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("loc", [0.0, 1.0])
@pytest.mark.parametrize("concentration", [0.1, 1.0, 10.0])
def test_von_mises_sample_log_prob(batch_shape, loc, concentration):
    loc_tensor = torch.full(batch_shape, loc)
    concentration_tensor = torch.full(batch_shape, concentration)

    dist = TensorVonMises(loc_tensor, concentration_tensor)
    torch_dist = TorchVonMises(loc_tensor, concentration_tensor)

    _check_sample_log_prob(dist, torch_dist)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("loc", [0.0, 1.0])
@pytest.mark.parametrize("concentration", [0.1, 1.0, 10.0])
def test_von_mises_compile(batch_shape, loc, concentration):
    loc_tensor = torch.full(batch_shape, loc)
    concentration_tensor = torch.full(batch_shape, concentration)

    compiled_dist_fn = compile_module(TensorVonMises)

    # Create a compiled instance
    compiled_dist = compiled_dist_fn(loc_tensor, concentration_tensor)

    # Check properties
    assert compiled_dist.loc.shape == loc_tensor.shape
    assert compiled_dist.concentration.shape == concentration_tensor.shape
    assert compiled_dist.batch_shape == batch_shape

    # Check sample and log_prob
    sample = compiled_dist.sample()
    assert sample.shape == batch_shape
    log_prob = compiled_dist.log_prob(sample)
    assert log_prob.shape == batch_shape

    # Check dist() method
    compiled_torch_dist = compiled_dist.dist()
    assert isinstance(compiled_torch_dist, TorchVonMises)
    assert compiled_torch_dist.loc.shape == loc_tensor.shape
    assert compiled_torch_dist.concentration.shape == concentration_tensor.shape
