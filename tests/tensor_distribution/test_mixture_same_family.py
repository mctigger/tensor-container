import pytest
import torch
from torch.distributions import MixtureSameFamily

from tensorcontainer.tensor_distribution.categorical import TensorCategorical
from tensorcontainer.tensor_distribution.mixture_same_family import (
    TensorMixtureSameFamily,
)
from tensorcontainer.tensor_distribution.normal import TensorNormal
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorMixtureSameFamily:
    def test_init_signatures_match(self):
        assert_init_signatures_match(TensorMixtureSameFamily, MixtureSameFamily)

    def test_properties_match(self):
        assert_properties_signatures_match(TensorMixtureSameFamily, MixtureSameFamily)

    def test_property_values_match(self):
        mixture_logits = torch.randn(3, 5)
        mixture_distribution = TensorCategorical(logits=mixture_logits)

        # component_distribution's batch_shape should be (*mixture_distribution.batch_shape, num_components)
        # and event_shape should be () for this test.
        # mixture_distribution.batch_shape is (3,)
        # num_components is 5 (from mixture_logits.shape[-1])
        component_loc = torch.randn(3, 5)
        component_scale = torch.rand(3, 5).exp()
        component_distribution = TensorNormal(loc=component_loc, scale=component_scale)

        td_mixture = TensorMixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        assert_property_values_match(td_mixture)

    @pytest.mark.parametrize(
        "mixture_shape, num_components",
        [
            ((), 2),
            ((5,), 3),
            ((3, 5), 4),
        ],
    )
    def test_broadcasting_shapes(self, mixture_shape, num_components):
        mixture_logits = torch.randn(*mixture_shape, num_components)
        mixture_distribution = TensorCategorical(logits=mixture_logits)

        component_loc = torch.randn(*mixture_shape, num_components)
        component_scale = torch.rand(*mixture_shape, num_components).exp()
        component_distribution = TensorNormal(loc=component_loc, scale=component_scale)

        td_mixture = TensorMixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        assert td_mixture.batch_shape == mixture_shape
        assert td_mixture.dist().batch_shape == mixture_shape

    @pytest.mark.parametrize(
        "mixture_shape, num_components",
        [
            ((), 2),
            ((5,), 3),
            ((3, 5), 4),
        ],
    )
    def test_compile_compatibility(self, mixture_shape, num_components):
        mixture_logits = torch.randn(*mixture_shape, num_components)
        mixture_distribution = TensorCategorical(logits=mixture_logits)

        component_loc = torch.randn(*mixture_shape, num_components)
        component_scale = torch.rand(*mixture_shape, num_components).exp()
        component_distribution = TensorNormal(loc=component_loc, scale=component_scale)

        td_mixture = TensorMixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        sample = td_mixture.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_mixture, fullgraph=False)
        # MixtureSameFamily does not support rsample
        # run_and_compare_compiled(rsample_fn, td_mixture, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_mixture, sample, fullgraph=False)
