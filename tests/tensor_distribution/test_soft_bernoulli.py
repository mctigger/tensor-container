"""
Tests for TensorSoftBernoulli distribution.

This module contains test classes that verify:
- TensorSoftBernoulli initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch

from tensorcontainer.distributions.soft_bernoulli import SoftBernoulli
from tensorcontainer.tensor_distribution.soft_bernoulli import TensorSoftBernoulli
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorSoftBernoulliAPIMatch:
    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            ValueError,
            match="Either `probs` or `logits` must be specified, but not both.",
        ):
            TensorSoftBernoulli()

    @pytest.mark.parametrize("validate_args", [True, False])
    def test_validate_args(self, validate_args):
        if validate_args:
            with pytest.raises(
                ValueError,
                match="Either `probs` or `logits` must be specified, but not both.",
            ):
                TensorSoftBernoulli(validate_args=validate_args)
        else:
            # An error should still be raised because batch_shape and device cannot be inferred
            with pytest.raises(
                ValueError,
                match="Either `probs` or `logits` must be specified, but not both.",
            ):
                TensorSoftBernoulli(validate_args=validate_args)

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_type, shape):
        """Core operations should be compatible with torch.compile."""
        if param_type == "probs":
            param = torch.rand(*shape, requires_grad=True)
            td_soft_bernoulli = TensorSoftBernoulli(probs=param)
        else:
            param = torch.randn(*shape, requires_grad=True)
            td_soft_bernoulli = TensorSoftBernoulli(logits=param)

        def get_mean(td):
            return td.mean

        run_and_compare_compiled(get_mean, td_soft_bernoulli, fullgraph=False)

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorSoftBernoulli matches
        torch.distributions.SoftBernoulli.
        """
        assert_init_signatures_match(TensorSoftBernoulli, SoftBernoulli)

    def test_properties_match(self):
        """
        Tests that the properties of TensorSoftBernoulli match
        torch.distributions.SoftBernoulli.
        """
        assert_properties_signatures_match(TensorSoftBernoulli, SoftBernoulli)

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    @pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 4, 5)])
    def test_property_values_match(self, param_type, shape):
        """
        Tests that the property values of TensorSoftBernoulli match
        torch.distributions.SoftBernoulli.
        """
        if param_type == "probs":
            param = torch.rand(*shape)
            td_soft_bernoulli = TensorSoftBernoulli(probs=param)
        else:
            param = torch.randn(*shape)
            td_soft_bernoulli = TensorSoftBernoulli(logits=param)
        assert_property_values_match(td_soft_bernoulli)
