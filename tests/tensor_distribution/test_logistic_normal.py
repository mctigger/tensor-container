import pytest
import torch
from torch.distributions import LogisticNormal as TorchLogisticNormal

from tensorcontainer.tensor_distribution.logistic_normal import TensorLogisticNormal
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorLogisticNormalAPIMatch:
    """
    Tests that the TensorLogisticNormal API matches the PyTorch LogisticNormal API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorLogisticNormal matches
        torch.distributions.LogisticNormal.
        """
        assert_init_signatures_match(TensorLogisticNormal, TorchLogisticNormal)

    def test_properties_match(self):
        """
        Tests that the properties of TensorLogisticNormal match
        torch.distributions.LogisticNormal.
        """
        assert_properties_signatures_match(TensorLogisticNormal, TorchLogisticNormal)

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
        Tests that the property values of TensorLogisticNormal match
        torch.distributions.LogisticNormal.
        """
        td_logistic_normal = TensorLogisticNormal(loc=loc, scale=scale)
        assert_property_values_match(td_logistic_normal)

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
        td_logistic_normal = TensorLogisticNormal(loc=loc, scale=scale)

        sample = td_logistic_normal.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_logistic_normal, fullgraph=False)
        run_and_compare_compiled(
            log_prob_fn, td_logistic_normal, sample, fullgraph=False
        )
