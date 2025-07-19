"""
Tests for ContinuousBernoulli distribution.

This module contains test classes that verify:
- ContinuousBernoulli initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from torch.distributions import ContinuousBernoulli as TorchContinuousBernoulli

from tensorcontainer.tensor_distribution.continuous_bernoulli import (
    TensorContinuousBernoulli,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestContinuousBernoulliAPIMatch:
    def test_init_no_params_raises_error(self):
        """A ValueError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            ValueError, match="Either `probs` or `logits` must be specified."
        ):
            TensorContinuousBernoulli()

    def test_init_mutually_exclusive_params_raises_error(self):
        """A ValueError should be raised when both probs and logits are provided."""
        with pytest.raises(
            ValueError,
            match="Either `probs` or `logits` must be specified, but not both.",
        ):
            TensorContinuousBernoulli(probs=torch.tensor(0.5), logits=torch.tensor(0.0))

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("property_name", ["mean", "variance", "probs", "logits"])
    def test_compile_compatibility(self, param_type, shape, property_name):
        """Core operations and properties should be compatible with torch.compile."""
        if param_type == "probs":
            param = torch.rand(*shape, requires_grad=True)
            td_dist = TensorContinuousBernoulli(probs=param)
        else:
            param = torch.randn(*shape, requires_grad=True)
            td_dist = TensorContinuousBernoulli(logits=param)

        def get_property(td):
            return getattr(td, property_name)

        run_and_compare_compiled(get_property, td_dist, fullgraph=False)

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of ContinuousBernoulli matches
        torch.distributions.ContinuousBernoulli.
        """
        assert_init_signatures_match(
            TensorContinuousBernoulli, TorchContinuousBernoulli
        )

    def test_properties_match(self):
        """
        Tests that the properties of ContinuousBernoulli match
        torch.distributions.ContinuousBernoulli.
        """
        assert_properties_signatures_match(
            TensorContinuousBernoulli, TorchContinuousBernoulli
        )

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_property_values_match(self, param_type, shape):
        """
        Tests that the property values of ContinuousBernoulli match
        torch.distributions.ContinuousBernoulli.
        """
        if param_type == "probs":
            param = torch.rand(*shape)
            td_dist = TensorContinuousBernoulli(probs=param)
        else:
            param = torch.randn(*shape)
            td_dist = TensorContinuousBernoulli(logits=param)
        assert_property_values_match(td_dist)
