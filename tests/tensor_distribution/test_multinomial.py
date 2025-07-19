"""
Tests for TensorMultinomial distribution.

This module contains test classes that verify:
- TensorMultinomial initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from torch.distributions import Multinomial

from tensorcontainer.tensor_distribution.multinomial import TensorMultinomial
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorMultinomialAPIMatch:
    def test_init_both_params_raises_error(self):
        """A RuntimeError should be raised when both probs and logits are provided."""
        with pytest.raises(
            RuntimeError, match="Only one of 'probs' or 'logits' can be provided."
        ):
            TensorMultinomial(probs=torch.rand(5), logits=torch.randn(5))

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("total_count", [1, 10])
    def test_compile_compatibility(self, param_type, shape, total_count):
        """Core operations should be compatible with torch.compile."""
        if param_type == "probs":
            param = torch.rand(*shape, requires_grad=True)
            td_multinomial = TensorMultinomial(total_count=total_count, probs=param)
        else:
            param = torch.randn(*shape, requires_grad=True)
            td_multinomial = TensorMultinomial(total_count=total_count, logits=param)

        def get_mean(td):
            return td.mean

        run_and_compare_compiled(get_mean, td_multinomial, fullgraph=False)

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorMultinomial matches
        torch.distributions.Multinomial.
        """
        assert_init_signatures_match(TensorMultinomial, Multinomial)

    def test_properties_match(self):
        """
        Tests that the properties of TensorMultinomial match
        torch.distributions.Multinomial.
        """
        assert_properties_signatures_match(TensorMultinomial, Multinomial)

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("total_count", [1, 10])
    def test_property_values_match(self, param_type, shape, total_count):
        """
        Tests that the property values of TensorMultinomial match
        torch.distributions.Multinomial.
        """
        if param_type == "probs":
            param = torch.rand(*shape)
            td_multinomial = TensorMultinomial(total_count=total_count, probs=param)
        else:
            param = torch.randn(*shape)
            td_multinomial = TensorMultinomial(total_count=total_count, logits=param)
        assert_property_values_match(td_multinomial)
