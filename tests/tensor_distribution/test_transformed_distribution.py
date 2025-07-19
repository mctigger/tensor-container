import pytest
import torch
from torch.distributions import (
    TransformedDistribution as TorchTransformedDistribution,
)
from torch.distributions.transforms import AffineTransform, ExpTransform

from tensorcontainer.tensor_distribution.normal import TensorNormal
from tensorcontainer.tensor_distribution.transformed_distribution import (
    TransformedDistribution,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_property_values_match,
)


class TestTransformedDistributionInitialization:
    @pytest.mark.parametrize(
        "loc_shape, scale_shape, expected_batch_shape",
        [
            ((), (), ()),
            ((5,), (), (5,)),
            ((), (5,), (5,)),
            ((3, 5), (5,), (3, 5)),
            ((5,), (3, 5), (3, 5)),
            ((2, 4, 5), (5,), (2, 4, 5)),
            ((5,), (2, 4, 5), (2, 4, 5)),
            ((2, 4, 5), (2, 4, 5), (2, 4, 5)),
        ],
    )
    def test_broadcasting_shapes(self, loc_shape, scale_shape, expected_batch_shape):
        loc = torch.randn(loc_shape)
        scale = torch.rand(scale_shape).exp()
        base_dist = TensorNormal(loc=loc, scale=scale)
        transforms = [ExpTransform()]
        td = TransformedDistribution(base_distribution=base_dist, transforms=transforms)
        assert td.batch_shape == expected_batch_shape
        assert td.dist().batch_shape == expected_batch_shape

    def test_dist_method(self):
        loc = torch.randn(1)
        scale = torch.rand(1).exp()
        base_dist = TensorNormal(loc=loc, scale=scale)
        transforms = [ExpTransform(), AffineTransform(loc=1.0, scale=2.0)]
        td = TransformedDistribution(base_distribution=base_dist, transforms=transforms)

        torch_td = td.dist()
        assert isinstance(torch_td, TorchTransformedDistribution)
        assert isinstance(torch_td.base_dist, type(base_dist.dist()))
        assert torch.allclose(torch_td.base_dist.loc, base_dist.dist().loc)
        assert torch.allclose(torch_td.base_dist.scale, base_dist.dist().scale)
        assert torch_td.transforms == transforms


class TestTransformedDistributionOperations:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        loc = torch.randn(*param_shape)
        scale = torch.rand(*param_shape).exp()
        base_dist = TensorNormal(loc=loc, scale=scale)
        transforms = [ExpTransform()]
        td = TransformedDistribution(base_distribution=base_dist, transforms=transforms)

        sample = td.sample()

        def sample_fn(td_instance):
            return td_instance.sample()

        def rsample_fn(td_instance):
            return td_instance.rsample()

        def log_prob_fn(td_instance, s):
            return td_instance.log_prob(s)

        run_and_compare_compiled(sample_fn, td, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td, sample, fullgraph=False)

    def test_sample_log_prob(self):
        loc = torch.randn(3, 5)
        scale = torch.rand(3, 5).exp()
        base_dist = TensorNormal(loc=loc, scale=scale)
        transforms = [ExpTransform()]
        td = TransformedDistribution(base_distribution=base_dist, transforms=transforms)

        torch_td = TorchTransformedDistribution(base_dist.dist(), transforms)

        sample_shape = (2, 1)
        sample = td.sample(sample_shape)
        assert sample.shape == torch_td.sample(sample_shape).shape
        assert torch.allclose(td.log_prob(sample), torch_td.log_prob(sample))

        rsample = td.rsample(sample_shape)
        assert rsample.shape == torch_td.rsample(sample_shape).shape
        assert torch.allclose(td.log_prob(rsample), torch_td.log_prob(rsample))


class TestTransformedDistributionAPIMatch:
    def test_properties_match(self):
        loc = torch.randn(3, 5)
        scale = torch.rand(3, 5).exp()
        base_dist = TensorNormal(loc=loc, scale=scale)
        transforms = [ExpTransform()]
        td = TransformedDistribution(base_distribution=base_dist, transforms=transforms)
        assert_property_values_match(td)
