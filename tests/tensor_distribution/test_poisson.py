"""
Tests for TensorPoisson distribution.

This module contains test classes that verify:
- TensorPoisson initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from torch.distributions import Poisson

from tensorcontainer.tensor_distribution.poisson import TensorPoisson
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorPoissonAPIMatch:
    @pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 4, 1)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        rate = torch.rand(*shape, requires_grad=True) + 0.1
        td_poisson = TensorPoisson(rate=rate)

        def get_mean(td):
            return td.mean

        run_and_compare_compiled(get_mean, td_poisson, fullgraph=False)

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorPoisson matches
        torch.distributions.Poisson.
        """
        assert_init_signatures_match(TensorPoisson, Poisson)

    def test_properties_match(self):
        """
        Tests that the properties of TensorPoisson match
        torch.distributions.Poisson.
        """
        assert_properties_signatures_match(TensorPoisson, Poisson)

    @pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 4, 1)])
    def test_property_values_match(self, shape):
        """
        Tests that the property values of TensorPoisson match
        torch.distributions.Poisson.
        """
        rate = torch.rand(*shape) + 0.1
        td_poisson = TensorPoisson(rate=rate)
        assert_property_values_match(td_poisson)
