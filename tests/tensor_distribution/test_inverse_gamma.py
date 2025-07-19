import pytest
import torch
from torch.distributions import InverseGamma

from tensorcontainer.tensor_distribution.inverse_gamma import TensorInverseGamma
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorInverseGammaInitialization:
    @pytest.mark.parametrize(
        "concentration, rate",
        [
            (1.0, 1.0),
            (torch.tensor(1.0), torch.tensor(1.0)),
            (torch.randn(2, 3).exp(), torch.randn(2, 3).exp()),
        ],
    )
    def test_init(self, concentration, rate):
        td_inverse_gamma = TensorInverseGamma(concentration=concentration, rate=rate)
        assert isinstance(td_inverse_gamma, TensorInverseGamma)
        assert isinstance(td_inverse_gamma.dist(), InverseGamma)


class TestTensorInverseGammaTensorContainerIntegration:
    @pytest.mark.parametrize(
        "concentration, rate",
        [
            (torch.randn(2, 3).exp(), torch.randn(2, 3).exp()),
        ],
    )
    def test_compile_compatibility(self, concentration, rate):
        """Core operations should be compatible with torch.compile."""
        td_inverse_gamma = TensorInverseGamma(concentration=concentration, rate=rate)

        sample = td_inverse_gamma.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_inverse_gamma, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_inverse_gamma, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_inverse_gamma, sample, fullgraph=False)


class TestTensorInverseGammaAPIMatch:
    """
    Tests that the TensorInverseGamma API matches the PyTorch InverseGamma API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorInverseGamma matches
        torch.distributions.InverseGamma.
        """
        assert_init_signatures_match(TensorInverseGamma, InverseGamma)

    def test_properties_match(self):
        """
        Tests that the properties of TensorInverseGamma match
        torch.distributions.InverseGamma.
        """
        assert_properties_signatures_match(TensorInverseGamma, InverseGamma)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorInverseGamma match
        torch.distributions.InverseGamma.
        """
        concentration = torch.randn(3, 5).exp()
        rate = torch.randn(3, 5).exp()
        td_inverse_gamma = TensorInverseGamma(concentration=concentration, rate=rate)
        assert_property_values_match(td_inverse_gamma)
