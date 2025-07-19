import pytest
import torch
from torch.distributions import Laplace

from tensorcontainer.tensor_distribution.laplace import TensorLaplace
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorLaplaceInitialization:
    @pytest.mark.parametrize(
        "loc, scale",
        [
            (0.0, 1.0),
            (torch.tensor(0.0), torch.tensor(1.0)),
            (torch.randn(2, 3), torch.randn(2, 3).exp()),
        ],
    )
    def test_init(self, loc, scale):
        td_laplace = TensorLaplace(loc=loc, scale=scale)
        assert isinstance(td_laplace, TensorLaplace)
        assert isinstance(td_laplace.dist(), Laplace)


class TestTensorLaplaceTensorContainerIntegration:
    @pytest.mark.parametrize(
        "loc, scale",
        [
            (torch.randn(2, 3), torch.randn(2, 3).exp()),
        ],
    )
    def test_compile_compatibility(self, loc, scale):
        """Core operations should be compatible with torch.compile."""
        td_laplace = TensorLaplace(loc=loc, scale=scale)

        sample = td_laplace.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_laplace, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_laplace, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_laplace, sample, fullgraph=False)


class TestTensorLaplaceAPIMatch:
    """
    Tests that the TensorLaplace API matches the PyTorch Laplace API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorLaplace matches
        torch.distributions.Laplace.
        """
        assert_init_signatures_match(TensorLaplace, Laplace)

    def test_properties_match(self):
        """
        Tests that the properties of TensorLaplace match
        torch.distributions.Laplace.
        """
        assert_properties_signatures_match(TensorLaplace, Laplace)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorLaplace match
        torch.distributions.Laplace.
        """
        loc = torch.randn(3, 5)
        scale = torch.randn(3, 5).exp()
        td_laplace = TensorLaplace(loc=loc, scale=scale)
        assert_property_values_match(td_laplace)
