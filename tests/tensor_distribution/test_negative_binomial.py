"""
Tests for TensorNegativeBinomial distribution.

This module contains test classes that verify:
- TensorNegativeBinomial initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from torch.distributions import NegativeBinomial

from tensorcontainer.tensor_distribution.negative_binomial import TensorNegativeBinomial
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorNegativeBinomialAPIMatch:
    @pytest.mark.parametrize("total_count_type", ["int", "tensor"])
    def test_init_total_count_non_positive_raises_error(self, total_count_type):
        """A ValueError should be raised when total_count is non-positive."""
        if total_count_type == "int":
            total_count = 0
        else:
            total_count = torch.tensor(0.0)
        with pytest.raises(ValueError, match="total_count must be positive"):
            TensorNegativeBinomial(total_count=total_count, probs=torch.rand(5))

    def test_init_both_params_raises_error(self):
        """A ValueError should be raised when both probs and logits are provided."""
        with pytest.raises(
            ValueError, match="Only one of 'probs' or 'logits' can be specified."
        ):
            TensorNegativeBinomial(
                total_count=1, probs=torch.rand(5), logits=torch.randn(5)
            )

    def test_init_no_params_raises_error(self):
        """A ValueError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            ValueError, match="Either 'probs' or 'logits' must be provided."
        ):
            TensorNegativeBinomial(total_count=1)

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("total_count", [1, 10])
    def test_compile_compatibility(self, param_type, shape, total_count):
        """Core operations should be compatible with torch.compile."""
        if param_type == "probs":
            param = torch.rand(*shape, requires_grad=True)
            td_negative_binomial = TensorNegativeBinomial(
                total_count=total_count, probs=param
            )
        else:
            param = torch.randn(*shape, requires_grad=True)
            td_negative_binomial = TensorNegativeBinomial(
                total_count=total_count, logits=param
            )

        def get_mean(td):
            return td.mean

        run_and_compare_compiled(get_mean, td_negative_binomial, fullgraph=False)

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorNegativeBinomial matches
        torch.distributions.NegativeBinomial.
        """
        assert_init_signatures_match(TensorNegativeBinomial, NegativeBinomial)

    def test_properties_match(self):
        """
        Tests that the properties of TensorNegativeBinomial match
        torch.distributions.NegativeBinomial.
        """
        assert_properties_signatures_match(TensorNegativeBinomial, NegativeBinomial)

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("total_count", [1, 10])
    def test_property_values_match(self, param_type, shape, total_count):
        """
        Tests that the property values of TensorNegativeBinomial match
        torch.distributions.NegativeBinomial.
        """
        if param_type == "probs":
            param = torch.rand(*shape)
            td_negative_binomial = TensorNegativeBinomial(
                total_count=total_count, probs=param
            )
        else:
            param = torch.randn(*shape)
            td_negative_binomial = TensorNegativeBinomial(
                total_count=total_count, logits=param
            )
        assert_property_values_match(td_negative_binomial)
