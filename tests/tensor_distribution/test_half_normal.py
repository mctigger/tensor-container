import pytest
import torch
from torch.distributions import HalfNormal

from tensorcontainer.tensor_distribution.half_normal import TensorHalfNormal
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorHalfNormalInitialization:
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

        td_half_normal = TensorHalfNormal(scale=scale)
        assert td_half_normal.batch_shape == expected_batch_shape
        assert td_half_normal.dist().batch_shape == expected_batch_shape

    def test_scalar_parameters(self):
        """Test initialization with scalar parameters."""
        scale = torch.tensor(1.0)
        td_half_normal = TensorHalfNormal(scale=scale)
        assert td_half_normal.batch_shape == ()
        assert td_half_normal.device == scale.device


class TestTensorHalfNormalTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        scale = torch.rand(*param_shape) + 0.1
        td_half_normal = TensorHalfNormal(scale=scale)

        sample = td_half_normal.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_half_normal, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_half_normal, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_half_normal, sample, fullgraph=False)

    def test_pytree_integration(self):
        """
        We use the copy method as a proxy to ensure pytree integration (e.g. unflattening)
        works correctly.
        """
        scale = torch.rand(3, 5) + 0.1
        original_dist = TensorHalfNormal(scale=scale)
        copied_dist = original_dist.copy()

        # Assert that it's a new instance
        assert copied_dist is not original_dist
        assert isinstance(copied_dist, TensorHalfNormal)


class TestTensorHalfNormalAPIMatch:
    """
    Tests that the TensorHalfNormal API matches the PyTorch HalfNormal API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorHalfNormal matches
        torch.distributions.HalfNormal.
        """
        assert_init_signatures_match(TensorHalfNormal, HalfNormal)

    def test_properties_match(self):
        """
        Tests that the properties of TensorHalfNormal match
        torch.distributions.HalfNormal.
        """
        assert_properties_signatures_match(TensorHalfNormal, HalfNormal)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorHalfNormal match
        torch.distributions.HalfNormal.
        """
        scale = torch.rand(3, 5) + 0.1
        td_half_normal = TensorHalfNormal(scale=scale)
        assert_property_values_match(td_half_normal)
