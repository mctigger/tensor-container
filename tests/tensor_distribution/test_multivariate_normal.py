"""
Tests for TensorMultivariateNormal distribution.

This module contains test classes that verify:
- TensorMultivariateNormal initialization and parameter validation
- Core distribution operations (sample, rsample, log_prob)
- TensorContainer integration (view, reshape, device operations)
- Distribution-specific properties and edge cases
"""

import pytest
import torch
from torch.distributions import MultivariateNormal

from tensorcontainer.tensor_distribution.multivariate_normal import (
    TensorMultivariateNormal,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorMultivariateNormalAPIMatch:
    @pytest.mark.parametrize(
        "params",
        [
            {"covariance_matrix": torch.eye(2), "precision_matrix": torch.eye(2)},
            {"covariance_matrix": torch.eye(2), "scale_tril": torch.eye(2)},
            {"precision_matrix": torch.eye(2), "scale_tril": torch.eye(2)},
            {
                "covariance_matrix": torch.eye(2),
                "precision_matrix": torch.eye(2),
                "scale_tril": torch.eye(2),
            },
        ],
    )
    def test_init_multiple_params_raises_error(self, params):
        """A ValueError should be raised when more than one of covariance_matrix, precision_matrix, or scale_tril are provided."""
        with pytest.raises(
            ValueError,
            match="Expected exactly one of `covariance_matrix`, `precision_matrix`, `scale_tril` to be specified, but got",
        ):
            TensorMultivariateNormal(loc=torch.zeros(2), **params)

    @pytest.mark.parametrize(
        "param_type", ["covariance_matrix", "precision_matrix", "scale_tril"]
    )
    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    @pytest.mark.parametrize("event_dim", [1, 2, 3])
    def test_compile_compatibility(self, param_type, batch_shape, event_dim):
        """Core operations should be compatible with torch.compile."""
        loc = torch.zeros(*batch_shape, event_dim, requires_grad=True)
        if param_type == "covariance_matrix":
            param = (
                torch.eye(event_dim)
                .expand(*batch_shape, event_dim, event_dim)
                .clone()
                .detach()
                .requires_grad_(True)
            )
            td_dist = TensorMultivariateNormal(loc=loc, covariance_matrix=param)
        elif param_type == "precision_matrix":
            param = (
                torch.eye(event_dim)
                .expand(*batch_shape, event_dim, event_dim)
                .clone()
                .detach()
                .requires_grad_(True)
            )
            td_dist = TensorMultivariateNormal(loc=loc, precision_matrix=param)
        else:  # scale_tril
            param = (
                torch.eye(event_dim)
                .expand(*batch_shape, event_dim, event_dim)
                .clone()
                .detach()
                .requires_grad_(True)
            )
            td_dist = TensorMultivariateNormal(loc=loc, scale_tril=param)

        def get_mean(td):
            return td.mean

        run_and_compare_compiled(get_mean, td_dist, fullgraph=False)

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorMultivariateNormal matches
        torch.distributions.MultivariateNormal.
        """
        assert_init_signatures_match(TensorMultivariateNormal, MultivariateNormal)

    def test_properties_match(self):
        """
        Tests that the properties of TensorMultivariateNormal match
        torch.distributions.MultivariateNormal.
        """
        assert_properties_signatures_match(TensorMultivariateNormal, MultivariateNormal)

    @pytest.mark.parametrize(
        "param_type", ["covariance_matrix", "precision_matrix", "scale_tril"]
    )
    @pytest.mark.parametrize("batch_shape", [(), (2,), (3, 2)])
    @pytest.mark.parametrize("event_dim", [1, 2, 3])
    def test_property_values_match(self, param_type, batch_shape, event_dim):
        """
        Tests that the property values of TensorMultivariateNormal match
        torch.distributions.MultivariateNormal.
        """
        loc = torch.zeros(*batch_shape, event_dim)
        if param_type == "covariance_matrix":
            param = torch.eye(event_dim).expand(*batch_shape, event_dim, event_dim)
            td_dist = TensorMultivariateNormal(loc=loc, covariance_matrix=param)
        elif param_type == "precision_matrix":
            param = torch.eye(event_dim).expand(*batch_shape, event_dim, event_dim)
            td_dist = TensorMultivariateNormal(loc=loc, precision_matrix=param)
        else:  # scale_tril
            param = torch.eye(event_dim).expand(*batch_shape, event_dim, event_dim)
            td_dist = TensorMultivariateNormal(loc=loc, scale_tril=param)
        assert_property_values_match(td_dist)
