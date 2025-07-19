"""
Tests for TensorNormal distribution.

This module contains test classes that verify:
- TensorNormal initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from torch.distributions import Normal

from tensorcontainer.tensor_distribution.normal import TensorNormal
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorNormalInitialization:
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
        """Test that batch_shape is correctly determined by broadcasting."""
        loc = torch.randn(loc_shape)
        scale = torch.rand(scale_shape).exp()  # scale must be positive
        td_normal = TensorNormal(loc=loc, scale=scale)
        assert td_normal.batch_shape == expected_batch_shape
        assert td_normal.dist().batch_shape == expected_batch_shape


class TestTensorNormalTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        loc = torch.randn(*param_shape)
        scale = torch.rand(*param_shape).exp()  # scale must be positive
        td_normal = TensorNormal(loc=loc, scale=scale)

        sample = td_normal.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_normal, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_normal, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_normal, sample, fullgraph=False)


class TestTensorNormalAPIMatch:
    """
    Tests that the TensorNormal API matches the PyTorch Normal API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorNormal matches
        torch.distributions.Normal.
        """
        assert_init_signatures_match(TensorNormal, Normal)

    def test_properties_match(self):
        """
        Tests that the properties of TensorNormal match
        torch.distributions.Normal.
        """
        assert_properties_signatures_match(TensorNormal, Normal)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorNormal match
        torch.distributions.Normal.
        """
        loc = torch.randn(3, 5)
        scale = torch.rand(3, 5).exp()  # scale must be positive
        td_normal = TensorNormal(loc=loc, scale=scale)
        assert_property_values_match(td_normal)
