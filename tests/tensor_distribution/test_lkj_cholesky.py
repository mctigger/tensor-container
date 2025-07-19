import pytest
import torch
from torch.distributions import LKJCholesky as TorchLKJCholesky

from tensorcontainer.tensor_distribution.lkj_cholesky import TensorLKJCholesky
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorLKJCholeskyAPIMatch:
    """
    Tests that the TensorLKJCholesky API matches the PyTorch LKJCholesky API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorLKJCholesky matches
        torch.distributions.LKJCholesky.
        """
        assert_init_signatures_match(TensorLKJCholesky, TorchLKJCholesky)

    def test_properties_match(self):
        """
        Tests that the properties of TensorLKJCholesky match
        torch.distributions.LKJCholesky.
        """
        assert_properties_signatures_match(TensorLKJCholesky, TorchLKJCholesky)

    @pytest.mark.parametrize(
        "dim, concentration",
        [
            (2, 1.0),
            (3, 0.5),
            (4, torch.tensor([1.0, 2.0])),
        ],
    )
    def test_property_values_match(self, dim, concentration):
        """
        Tests that the property values of TensorLKJCholesky match
        torch.distributions.LKJCholesky.
        """
        td_lkj = TensorLKJCholesky(dim=dim, concentration=concentration)
        assert_property_values_match(td_lkj)

    @pytest.mark.parametrize(
        "dim, concentration",
        [
            (2, 1.0),
            (3, 0.5),
            (4, torch.tensor([1.0, 2.0])),
        ],
    )
    def test_compile_compatibility(self, dim, concentration):
        """Core operations should be compatible with torch.compile."""
        td_lkj = TensorLKJCholesky(dim=dim, concentration=concentration)

        sample = td_lkj.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_lkj, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_lkj, sample, fullgraph=False)
