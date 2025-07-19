import pytest
import torch
from torch.distributions import (
    LowRankMultivariateNormal as TorchLowRankMultivariateNormal,
)

from tensorcontainer.tensor_distribution.low_rank_multivariate_normal import (
    TensorLowRankMultivariateNormal,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorLowRankMultivariateNormalTensorContainerIntegration:
    @pytest.mark.parametrize(
        "batch_shape, event_shape, rank",
        [
            ((), 2, 1),
            ((5,), 2, 1),
            ((3, 5), 2, 1),
            ((3, 5), 4, 2),
        ],
    )
    def test_compile_compatibility(self, batch_shape, event_shape, rank):
        """Core operations should be compatible with torch.compile."""
        loc = torch.randn(*batch_shape, event_shape)
        cov_factor = torch.randn(*batch_shape, event_shape, rank)
        cov_diag = torch.rand(*batch_shape, event_shape).exp()

        td_dist = TensorLowRankMultivariateNormal(
            loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
        )

        sample = td_dist.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_dist, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_dist, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_dist, sample, fullgraph=False)


class TestTensorLowRankMultivariateNormalAPIMatch:
    """
    Tests that the TensorLowRankMultivariateNormal API matches the PyTorch LowRankMultivariateNormal API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorLowRankMultivariateNormal matches
        torch.distributions.LowRankMultivariateNormal.
        """
        assert_init_signatures_match(
            TensorLowRankMultivariateNormal, TorchLowRankMultivariateNormal
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorLowRankMultivariateNormal match
        torch.distributions.LowRankMultivariateNormal.
        """
        assert_properties_signatures_match(
            TensorLowRankMultivariateNormal, TorchLowRankMultivariateNormal
        )

    def test_property_values_match(self):
        """
        Tests that the property values of TensorLowRankMultivariateNormal match
        torch.distributions.LowRankMultivariateNormal.
        """
        loc = torch.randn(3, 5)
        cov_factor = torch.randn(3, 5, 2)
        cov_diag = torch.rand(3, 5).exp()
        td_dist = TensorLowRankMultivariateNormal(
            loc=loc, cov_factor=cov_factor, cov_diag=cov_diag
        )
        assert_property_values_match(td_dist)
