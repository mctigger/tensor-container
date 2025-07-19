"""
Tests for TensorDirichlet distribution.

This module contains test classes that verify:
- TensorDirichlet initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing
from torch.distributions import Dirichlet

from tensorcontainer.tensor_distribution.dirichlet import TensorDirichlet
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorDirichlet:
    @pytest.fixture
    def concentration(self):
        return torch.rand(3, 5).exp()

    @pytest.fixture
    def td_dirichlet(self, concentration):
        return TensorDirichlet(concentration=concentration)

    @pytest.fixture
    def torch_dirichlet(self, concentration):
        return Dirichlet(concentration=concentration)

    def test_init_signatures_match(self):
        assert_init_signatures_match(TensorDirichlet, Dirichlet)

    def test_properties_match(self):
        assert_properties_signatures_match(TensorDirichlet, Dirichlet)

    def test_property_values_match(self, td_dirichlet):
        assert_property_values_match(td_dirichlet)

    @pytest.mark.parametrize("concentration_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, concentration_shape):
        concentration = torch.rand(*concentration_shape).exp()
        td_dirichlet = TensorDirichlet(concentration=concentration)
        sample = td_dirichlet.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_dirichlet, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_dirichlet, sample, fullgraph=False)
