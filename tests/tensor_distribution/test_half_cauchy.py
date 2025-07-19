import pytest
import torch
from torch.distributions import HalfCauchy

from tensorcontainer.tensor_distribution.half_cauchy import TensorHalfCauchy
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorHalfCauchyInitialization:
    @pytest.mark.parametrize(
        "param_shape, expected_batch_shape",
        [
            ((), ()),
            ((5,), (5,)),
            ((3, 5), (3, 5)),
            ((2, 4, 5), (2, 4, 5)),
        ],
    )
    def test_broadcasting_shapes(self, param_shape, expected_batch_shape):
        """Test that batch_shape is correctly determined by broadcasting."""
        scale = torch.rand(param_shape) + 0.1  # scale must be positive

        td_half_cauchy = TensorHalfCauchy(scale=scale)
        assert td_half_cauchy.batch_shape == expected_batch_shape
        assert td_half_cauchy.dist().batch_shape == expected_batch_shape

    def test_scalar_parameters(self):
        """Test initialization with scalar parameters."""
        scale = torch.tensor(1.0)
        td_half_cauchy = TensorHalfCauchy(scale=scale)
        assert td_half_cauchy.batch_shape == ()
        assert td_half_cauchy.device == scale.device


class TestTensorHalfCauchyTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        scale = torch.rand(*param_shape) + 0.1
        td_half_cauchy = TensorHalfCauchy(scale=scale)

        sample = td_half_cauchy.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_half_cauchy, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_half_cauchy, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_half_cauchy, sample, fullgraph=False)

    def test_pytree_integration(self):
        """
        We use the copy method as a proxy to ensure pytree integration (e.g. unflattening)
        works correctly.
        """
        scale = torch.rand(3, 5) + 0.1
        original_dist = TensorHalfCauchy(scale=scale)
        copied_dist = original_dist.copy()

        # Assert that it's a new instance
        assert copied_dist is not original_dist
        assert isinstance(copied_dist, TensorHalfCauchy)


class TestTensorHalfCauchyAPIMatch:
    """
    Tests that the TensorHalfCauchy API matches the PyTorch HalfCauchy API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorHalfCauchy matches
        torch.distributions.HalfCauchy.
        """
        assert_init_signatures_match(TensorHalfCauchy, HalfCauchy)

    def test_properties_match(self):
        """
        Tests that the properties of TensorHalfCauchy match
        torch.distributions.HalfCauchy.
        """
        assert_properties_signatures_match(TensorHalfCauchy, HalfCauchy)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorHalfCauchy match
        torch.distributions.HalfCauchy.
        """
        scale = torch.rand(3, 5) + 0.1
        td_half_cauchy = TensorHalfCauchy(scale=scale)
        assert_property_values_match(td_half_cauchy)
