"""
Tests for TensorRelaxedBernoulli distribution.

This module contains test classes that verify:
- TensorRelaxedBernoulli initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from torch.distributions import RelaxedBernoulli

from tensorcontainer.tensor_distribution.relaxed_bernoulli import TensorRelaxedBernoulli
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
)


class TestTensorRelaxedBernoulliAPIMatch:
    @pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 4, 1)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        temperature = torch.rand(*shape, requires_grad=True) + 0.1
        probs = torch.rand(*shape, requires_grad=True)
        td_relaxed_bernoulli = TensorRelaxedBernoulli(
            temperature=temperature, probs=probs
        )

        def get_temperature(td):
            return td.temperature

        run_and_compare_compiled(get_temperature, td_relaxed_bernoulli, fullgraph=False)

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorRelaxedBernoulli matches
        torch.distributions.RelaxedBernoulli.
        """
        assert_init_signatures_match(TensorRelaxedBernoulli, RelaxedBernoulli)

    def test_properties_match(self):
        """
        Tests that the properties of TensorRelaxedBernoulli match
        torch.distributions.RelaxedBernoulli.
        """
        assert_properties_signatures_match(TensorRelaxedBernoulli, RelaxedBernoulli)

    @pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 4, 1)])
    def test_property_values_match(self, shape):
        """
        Tests that the property values of TensorRelaxedBernoulli match
        torch.distributions.RelaxedBernoulli.
        """
        temperature = torch.rand(*shape) + 0.1
        probs = torch.rand(*shape)
        td_relaxed_bernoulli = TensorRelaxedBernoulli(
            temperature=temperature, probs=probs
        )
        assert td_relaxed_bernoulli.temperature.allclose(temperature)
        assert td_relaxed_bernoulli.probs is not None
        assert td_relaxed_bernoulli.probs.allclose(probs)
        assert td_relaxed_bernoulli.logits is not None
        assert td_relaxed_bernoulli.logits.allclose(
            RelaxedBernoulli(temperature=temperature, probs=probs).logits
        )
