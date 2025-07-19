import pytest
import torch
from torch.distributions import Kumaraswamy

from tensorcontainer.tensor_distribution.kumaraswamy import TensorKumaraswamy
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorKumaraswamyInitialization:
    @pytest.mark.parametrize(
        "concentration1, concentration0",
        [
            (1.0, 1.0),
            (torch.tensor(1.0), torch.tensor(1.0)),
            (torch.randn(2, 3).exp(), torch.randn(2, 3).exp()),
        ],
    )
    def test_init(self, concentration1, concentration0):
        td_kumaraswamy = TensorKumaraswamy(
            concentration1=concentration1, concentration0=concentration0
        )
        assert isinstance(td_kumaraswamy, TensorKumaraswamy)
        assert isinstance(td_kumaraswamy.dist(), Kumaraswamy)


class TestTensorKumaraswamyTensorContainerIntegration:
    @pytest.mark.parametrize(
        "concentration1, concentration0",
        [
            (torch.randn(2, 3).exp(), torch.randn(2, 3).exp()),
        ],
    )
    def test_compile_compatibility(self, concentration1, concentration0):
        """Core operations should be compatible with torch.compile."""
        td_kumaraswamy = TensorKumaraswamy(
            concentration1=concentration1, concentration0=concentration0
        )

        sample = td_kumaraswamy.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_kumaraswamy, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_kumaraswamy, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_kumaraswamy, sample, fullgraph=False)


class TestTensorKumaraswamyAPIMatch:
    """
    Tests that the TensorKumaraswamy API matches the PyTorch Kumaraswamy API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorKumaraswamy matches
        torch.distributions.Kumaraswamy.
        """
        assert_init_signatures_match(TensorKumaraswamy, Kumaraswamy)

    def test_properties_match(self):
        """
        Tests that the properties of TensorKumaraswamy match
        torch.distributions.Kumaraswamy.
        """
        assert_properties_signatures_match(TensorKumaraswamy, Kumaraswamy)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorKumaraswamy match
        torch.distributions.Kumaraswamy.
        """
        concentration1 = torch.randn(3, 5).exp()
        concentration0 = torch.randn(3, 5).exp()
        td_kumaraswamy = TensorKumaraswamy(
            concentration1=concentration1, concentration0=concentration0
        )
        assert_property_values_match(td_kumaraswamy)
