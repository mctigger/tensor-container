import pytest
import torch
from torch.distributions import LogNormal as TorchLogNormal

from tensorcontainer.tensor_distribution.log_normal import TensorLogNormal
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorLogNormalAPIMatch:
    """
    Tests that the TensorLogNormal API matches the PyTorch LogNormal API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorLogNormal matches
        torch.distributions.LogNormal.
        """
        assert_init_signatures_match(TensorLogNormal, TorchLogNormal)

    def test_properties_match(self):
        """
        Tests that the properties of TensorLogNormal match
        torch.distributions.LogNormal.
        """
        assert_properties_signatures_match(TensorLogNormal, TorchLogNormal)

    @pytest.mark.parametrize(
        "loc, scale",
        [
            (0.0, 1.0),
            (torch.tensor([0.0, 1.0]), 1.0),
            (0.0, torch.tensor([1.0, 2.0])),
            (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])),
        ],
    )
    def test_property_values_match(self, loc, scale):
        """
        Tests that the property values of TensorLogNormal match
        torch.distributions.LogNormal.
        """
        td_log_normal = TensorLogNormal(loc=loc, scale=scale)
        assert_property_values_match(td_log_normal)

    @pytest.mark.parametrize(
        "loc, scale",
        [
            (0.0, 1.0),
            (torch.tensor([0.0, 1.0]), 1.0),
            (0.0, torch.tensor([1.0, 2.0])),
            (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 2.0])),
        ],
    )
    def test_compile_compatibility(self, loc, scale):
        """Core operations should be compatible with torch.compile."""
        td_log_normal = TensorLogNormal(loc=loc, scale=scale)

        sample = td_log_normal.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_log_normal, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_log_normal, sample, fullgraph=False)
