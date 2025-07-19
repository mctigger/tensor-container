"""
Tests for TensorTruncatedNormal distribution.

This module contains test classes that verify:
- TensorTruncatedNormal initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch

from src.tensorcontainer.distributions.truncated_normal import TruncatedNormal
from src.tensorcontainer.tensor_distribution.truncated_normal import (
    TensorTruncatedNormal,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorTruncatedNormalInitialization:
    @pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
    @pytest.mark.parametrize("event_shape", [(), (1,), (2, 3)])
    def test_truncated_normal_init(self, batch_shape, event_shape, device):
        # Ensure that the shapes are correctly handled for scalar tensors
        loc_param_shape = batch_shape + event_shape
        loc = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
        scale = (
            torch.rand(loc_param_shape if loc_param_shape else (), device=device) + 1e-6
        )
        low = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
        high = (
            low
            + torch.rand(loc_param_shape if loc_param_shape else (), device=device)
            + 1e-6
        )

        dist = TensorTruncatedNormal(loc=loc, scale=scale, low=low, high=high)

        expected_batch_shape = loc.shape[:-1] if loc.ndim > 0 else torch.Size([])
        expected_event_shape = loc.shape[-1:] if loc.ndim > 0 else torch.Size([])

        assert dist.batch_shape == expected_batch_shape
        assert dist.event_shape == expected_event_shape
        assert dist.loc.shape == loc.shape
        assert dist.scale.shape == scale.shape
        assert dist.low.shape == low.shape
        assert dist.high.shape == high.shape


class TestTensorTruncatedNormalAPIMatch:
    def test_init_signatures_match(self, device):
        assert_init_signatures_match(TensorTruncatedNormal, TruncatedNormal)

    def test_properties_match(self, device):
        assert_properties_signatures_match(TensorTruncatedNormal, TruncatedNormal)

    def test_property_values_match(self, device):
        loc = torch.randn(3, 5, device=device)
        scale = torch.rand(3, 5, device=device) + 1e-6
        low = torch.randn(3, 5, device=device)
        high = low + torch.rand(3, 5, device=device) + 1e-6
        td_truncated_normal = TensorTruncatedNormal(
            loc=loc, scale=scale, low=low, high=high
        )
        assert_property_values_match(td_truncated_normal)


class TestTensorTruncatedNormalMethods:
    @pytest.mark.parametrize("batch_shape", [(), (1,), (2, 3)])
    @pytest.mark.parametrize("event_shape", [(), (1,), (2, 3)])
    def test_truncated_normal_methods(self, batch_shape, event_shape, device):
        loc_param_shape = batch_shape + event_shape
        loc = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
        scale = (
            torch.rand(loc_param_shape if loc_param_shape else (), device=device) + 1e-6
        )
        low = torch.randn(loc_param_shape if loc_param_shape else (), device=device)
        high = (
            low
            + torch.rand(loc_param_shape if loc_param_shape else (), device=device)
            + 1e-6
        )

        dist = TensorTruncatedNormal(loc=loc, scale=scale, low=low, high=high)

        # Test rsample
        sample = dist.rsample()
        assert sample.shape == dist.batch_shape + dist.event_shape

        # Test log_prob
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == dist.batch_shape


class TestTensorTruncatedNormalCompileCompatibility:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape, device):
        loc = torch.randn(*param_shape, device=device)
        scale = torch.rand(*param_shape, device=device) + 1e-6
        low = torch.randn(*param_shape, device=device)
        high = low + torch.rand(*param_shape, device=device) + 1e-6

        td_truncated_normal = TensorTruncatedNormal(
            loc=loc, scale=scale, low=low, high=high, validate_args=False
        )

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(rsample_fn, td_truncated_normal, fullgraph=False)
        value = td_truncated_normal.rsample(torch.Size((1,)))
        run_and_compare_compiled(
            log_prob_fn, td_truncated_normal, value, fullgraph=False
        )
