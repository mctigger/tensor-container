"""
Tests for TensorCauchy distribution.

This module contains test classes that verify:
- TensorCauchy initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing

from tensorcontainer.tensor_distribution.cauchy import TensorCauchy
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorCauchyInitialization:
    @pytest.mark.parametrize(
        "loc, scale",
        [
            (0.0, 1.0),
            (torch.tensor([0.0]), torch.tensor([1.0])),
            (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])),
            (torch.tensor([0.0]), torch.tensor([1.0, 2.0])),
            (torch.tensor([0.0, 1.0]), torch.tensor([1.0])),
        ],
    )
    def test_valid_initialization(self, loc, scale):
        """Test valid initializations."""
        dist = TensorCauchy(loc=loc, scale=scale)
        assert isinstance(dist, TensorCauchy)
        assert dist.loc is not None
        assert dist.scale is not None

    @pytest.mark.parametrize(
        "loc, scale, error_type, error_match",
        [
            (
                torch.tensor([0.0]),
                torch.tensor([-1.0]),
                ValueError,
                "(?s)Expected parameter scale.*to satisfy the constraint GreaterThan\\(lower_bound=0\\.0\\).*",
            ),
            (
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                ValueError,
                "(?s)Expected parameter scale.*to satisfy the constraint GreaterThan\\(lower_bound=0\\.0\\).*",
            ),
        ],
    )
    def test_invalid_initialization(self, loc, scale, error_type, error_match):
        """Test invalid initializations."""
        with pytest.raises(error_type, match=error_match):
            TensorCauchy(loc=loc, scale=scale)


class TestTensorCauchyTensorContainerIntegration:
    @pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 4, 1)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        loc = torch.randn(*shape, requires_grad=True)
        scale = torch.rand(*shape, requires_grad=True) + 0.1  # scale must be positive
        td_cauchy = TensorCauchy(loc=loc, scale=scale)
        sample = td_cauchy.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_cauchy, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_cauchy, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_cauchy, sample, fullgraph=False)


class TestTensorCauchyAPIMatch:
    """
    Tests that the TensorCauchy API matches the PyTorch Cauchy API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorCauchy matches
        torch.distributions.Cauchy.
        """
        assert_init_signatures_match(TensorCauchy, torch.distributions.Cauchy)

    def test_properties_match(self):
        """
        Tests that the properties of TensorCauchy match
        torch.distributions.Cauchy.
        """
        assert_properties_signatures_match(TensorCauchy, torch.distributions.Cauchy)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorCauchy match
        torch.distributions.Cauchy.
        """
        loc = torch.randn(3, 5)
        scale = torch.rand(3, 5) + 0.1
        td_cauchy = TensorCauchy(loc=loc, scale=scale)
        assert_property_values_match(td_cauchy)
