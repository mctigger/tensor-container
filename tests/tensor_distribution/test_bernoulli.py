"""
Tests for TensorBernoulli distribution.

This module contains test classes that verify:
- TensorBernoulli initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from torch.distributions import Bernoulli

from tensorcontainer.tensor_distribution.bernoulli import TensorBernoulli
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorBernoulliAPIMatch:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            ValueError,
            match="Either `probs` or `logits` must be specified, but not both.",
        ):
            TensorBernoulli()

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_type, shape):
        """Core operations should be compatible with torch.compile."""
        if param_type == "probs":
            param = torch.rand(*shape, requires_grad=True)
            td_bernoulli = TensorBernoulli(probs=param)
        else:
            param = torch.randn(*shape, requires_grad=True)
            td_bernoulli = TensorBernoulli(logits=param)

        def get_mean(td):
            return td.mean

        run_and_compare_compiled(get_mean, td_bernoulli, fullgraph=False)

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorBernoulli matches
        torch.distributions.Bernoulli.
        """
        assert_init_signatures_match(TensorBernoulli, Bernoulli)

    def test_properties_match(self):
        """
        Tests that the properties of TensorBernoulli match
        torch.distributions.Bernoulli.
        """
        assert_properties_signatures_match(TensorBernoulli, Bernoulli)

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_property_values_match(self, param_type, shape):
        """
        Tests that the property values of TensorBernoulli match
        torch.distributions.Bernoulli.
        """
        if param_type == "probs":
            param = torch.rand(*shape)
            td_bernoulli = TensorBernoulli(probs=param)
        else:
            param = torch.randn(*shape)
            td_bernoulli = TensorBernoulli(logits=param)
        assert_property_values_match(td_bernoulli)

    """
    Tests that the TensorBernoulli API matches the PyTorch Bernoulli API.
    """
