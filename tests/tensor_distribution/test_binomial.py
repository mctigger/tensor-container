"""
Tests for TensorBinomial distribution.

This module contains test classes that verify:
- TensorBinomial initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
import torch.distributions
import torch.testing

from tensorcontainer.tensor_distribution.binomial import TensorBinomial
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorBinomialInitialization:
    def test_init_no_params_raises_error(self):
        """A ValueError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            ValueError,
            match="Either `probs` or `logits` must be specified, but not both.",
        ):
            TensorBinomial(total_count=10)

    def test_init_with_probs(self):
        """Test initialization with probs."""
        dist = TensorBinomial(total_count=10, probs=torch.tensor([0.5]))
        assert isinstance(dist, TensorBinomial)
        torch.testing.assert_close(dist.total_count, torch.tensor([10.0]))
        torch.testing.assert_close(dist.probs, torch.tensor([0.5]))

    def test_init_with_logits(self):
        """Test initialization with logits."""
        dist = TensorBinomial(total_count=10, logits=torch.tensor([0.0]))
        assert isinstance(dist, TensorBinomial)
        torch.testing.assert_close(dist.total_count, torch.tensor([10.0]))
        torch.testing.assert_close(dist.logits, torch.tensor([0.0]))

    def test_init_with_tensor_total_count(self):
        """Test initialization with total_count as a Tensor."""
        total_count = torch.tensor([10, 20])
        probs = torch.tensor([0.5, 0.8])
        dist = TensorBinomial(total_count=total_count, probs=probs)
        assert isinstance(dist, TensorBinomial)
        torch.testing.assert_close(dist.total_count, total_count.float())
        torch.testing.assert_close(dist.probs, probs)

    def test_init_with_broadcasted_params(self):
        """Test initialization with broadcasted parameters."""
        total_count = torch.tensor([10, 20]).reshape(2, 1)
        probs = torch.tensor([0.5, 0.8]).reshape(1, 2)
        dist = TensorBinomial(total_count=total_count, probs=probs)
        assert isinstance(dist, TensorBinomial)
        expected_total_count = torch.tensor([[10, 10], [20, 20]]).float()
        expected_probs = torch.tensor([[0.5, 0.8], [0.5, 0.8]])
        torch.testing.assert_close(dist.total_count, expected_total_count)
        torch.testing.assert_close(dist.probs, expected_probs)


class TestTensorBinomialTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_compile_compatibility(self, param_shape, param_type):
        """Core operations should be compatible with torch.compile."""
        total_count = 10
        if param_type == "probs":
            param = torch.rand(*param_shape, requires_grad=True)
            td_binomial = TensorBinomial(total_count=total_count, probs=param)
        else:
            param = torch.randn(*param_shape, requires_grad=True)
            td_binomial = TensorBinomial(total_count=total_count, logits=param)

        sample = td_binomial.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            # Binomial does not support rsample
            with pytest.raises(NotImplementedError):
                return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_binomial, fullgraph=False)
        # run_and_compare_compiled(rsample_fn, td_binomial, fullgraph=False) # rsample not supported
        run_and_compare_compiled(log_prob_fn, td_binomial, sample, fullgraph=False)

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_copy_method(self, param_type):
        """Test that the .copy() method works correctly."""
        total_count = 10
        if param_type == "probs":
            param = torch.rand(3, 5)
            td_binomial = TensorBinomial(total_count=total_count, probs=param)
        else:
            param = torch.randn(3, 5)
            td_binomial = TensorBinomial(total_count=total_count, logits=param)

        td_binomial_copy = td_binomial.copy()

        assert isinstance(td_binomial_copy, TensorBinomial)
        torch.testing.assert_close(
            td_binomial.total_count, td_binomial_copy.total_count
        )
        if param_type == "probs":
            torch.testing.assert_close(td_binomial.probs, td_binomial_copy.probs)
        else:
            torch.testing.assert_close(td_binomial.logits, td_binomial_copy.logits)

        # Ensure they are different objects
        assert td_binomial is not td_binomial_copy
        assert td_binomial.total_count is not td_binomial_copy.total_count
        if param_type == "probs":
            assert td_binomial.probs is not td_binomial_copy.probs
        else:
            assert td_binomial.logits is not td_binomial_copy.logits


class TestTensorBinomialAPIMatch:
    """
    Tests that the TensorBinomial API matches the PyTorch Binomial API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorBinomial matches
        torch.distributions.Binomial.
        """
        assert_init_signatures_match(TensorBinomial, torch.distributions.Binomial)

    def test_properties_match(self):
        """
        Tests that the properties of TensorBinomial match
        torch.distributions.Binomial.
        """
        assert_properties_signatures_match(TensorBinomial, torch.distributions.Binomial)

    @pytest.mark.parametrize("total_count_type", ["int", "tensor"])
    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_property_values_match(self, total_count_type, param_type):
        """
        Tests that the property values of TensorBinomial match
        torch.distributions.Binomial.
        """
        if total_count_type == "int":
            total_count = 10
        else:
            # Make total_count a scalar tensor for broadcasting with multi-dimensional param
            total_count = torch.tensor(10)

        if param_type == "probs":
            param = torch.rand(3, 5)
            td_binomial = TensorBinomial(total_count=total_count, probs=param)
        else:
            param = torch.randn(3, 5)
            td_binomial = TensorBinomial(total_count=total_count, logits=param)
        assert_property_values_match(td_binomial)
