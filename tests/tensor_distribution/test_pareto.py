"""
Tests for TensorPareto distribution.

This module contains test classes that verify:
- TensorPareto initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from torch.distributions import Pareto

from tensorcontainer.tensor_distribution.pareto import TensorPareto
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorParetoAPIMatch:
    @pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 4, 1)])
    def test_compile_compatibility(self, shape):
        """Core operations should be compatible with torch.compile."""
        scale = torch.rand(*shape, requires_grad=True) + 0.1
        alpha = torch.rand(*shape, requires_grad=True) + 0.1
        td_pareto = TensorPareto(scale=scale, alpha=alpha)

        def get_mean(td):
            return td.mean

        run_and_compare_compiled(get_mean, td_pareto, fullgraph=False)

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorPareto matches
        torch.distributions.Pareto.
        """
        assert_init_signatures_match(TensorPareto, Pareto)

    def test_properties_match(self):
        """
        Tests that the properties of TensorPareto match
        torch.distributions.Pareto.
        """
        assert_properties_signatures_match(TensorPareto, Pareto)

    @pytest.mark.parametrize("shape", [(1,), (3, 1), (2, 4, 1)])
    def test_property_values_match(self, shape):
        """
        Tests that the property values of TensorPareto match
        torch.distributions.Pareto.
        """
        scale = torch.rand(*shape) + 0.1
        alpha = torch.rand(*shape) + 0.1
        td_pareto = TensorPareto(scale=scale, alpha=alpha)
        assert_property_values_match(td_pareto)
