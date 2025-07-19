import torch
from torch.distributions import Normal, TransformedDistribution

from src.tensorcontainer.tensor_distribution.tanh_normal import (
    ClampedTanhTransform,
    TensorTanhNormal,
)
from tests.compile_utils import run_and_compare_compiled


class TestTensorTanhNormal:
    def test_initialization(self):
        # Test with scalar loc and scale
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        dist = TensorTanhNormal(loc, scale)
        assert dist.loc == loc
        assert dist.scale == scale
        assert dist.shape == torch.Size([])
        assert dist.device == loc.device

        # Test with tensor loc and scale
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 2.0])
        dist = TensorTanhNormal(loc, scale)
        assert torch.equal(dist.loc, loc)
        assert torch.equal(dist.scale, scale)
        assert dist.shape == loc.shape
        assert dist.device == loc.device

        # Test with reinterpreted_batch_ndims
        loc = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        dist = TensorTanhNormal(loc, scale, reinterpreted_batch_ndims=1)
        assert torch.equal(dist.loc, loc)
        assert torch.equal(dist.scale, scale)
        assert dist.shape == loc.shape
        assert dist.device == loc.device
        assert dist._reinterpreted_batch_ndims == 1

    def test_rsample_and_log_prob(self):
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        tensor_dist = TensorTanhNormal(loc, scale)

        # Create a reference torch.distributions.TransformedDistribution
        base_normal = Normal(loc, scale)
        reference_dist_base = TransformedDistribution(
            base_normal, ClampedTanhTransform()
        )

        # Wrap the reference distribution in Independent to match TensorTanhNormal's behavior
        reference_dist = torch.distributions.Independent(reference_dist_base, 1)

        sample_shape = torch.Size((100,))
        samples = tensor_dist.rsample(sample_shape)
        assert samples.shape == sample_shape + loc.shape

        # Check log_prob
        log_probs = tensor_dist.log_prob(samples)
        reference_log_probs = reference_dist.log_prob(samples)
        assert torch.allclose(log_probs, reference_log_probs, atol=1e-5)

        # Check mean (approximate for rsample)
        # This is a sanity check, not a strict equality
        # Removed: TransformedDistribution does not implement .mean

    def test_compile_compatibility(self):
        loc = torch.tensor([0.0, 1.0])
        scale = torch.tensor([1.0, 0.5])
        dist = TensorTanhNormal(loc, scale)

        # Test rsample
        run_and_compare_compiled(dist.rsample, torch.Size((5,)))

        # Test log_prob
        value = dist.rsample(torch.Size((1,)))
        run_and_compare_compiled(dist.log_prob, value)
