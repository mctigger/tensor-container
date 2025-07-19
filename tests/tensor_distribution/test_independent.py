import pytest
import torch
from torch.distributions import Independent

from tensorcontainer.tensor_distribution.independent import TensorIndependent
from tensorcontainer.tensor_distribution.normal import TensorNormal
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorIndependentInitialization:
    @pytest.mark.parametrize(
        "base_dist_shape, reinterpreted_batch_ndims, expected_batch_shape",
        [
            ((), 0, ()),
            ((5,), 0, (5,)),
            ((5,), 1, ()),
            ((3, 5), 0, (3, 5)),
            ((3, 5), 1, (3,)),
            ((3, 5), 2, ()),
            ((2, 4, 5), 0, (2, 4, 5)),
            ((2, 4, 5), 1, (2, 4)),
            ((2, 4, 5), 2, (2,)),
            ((2, 4, 5), 3, ()),
        ],
    )
    def test_broadcasting_shapes(
        self, base_dist_shape, reinterpreted_batch_ndims, expected_batch_shape
    ):
        """Test that batch_shape is correctly determined by broadcasting."""
        loc = torch.randn(base_dist_shape)
        scale = torch.rand(base_dist_shape).exp()
        base_distribution = TensorNormal(loc=loc, scale=scale)
        td_independent = TensorIndependent(
            base_distribution=base_distribution,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
        assert td_independent.batch_shape == expected_batch_shape
        assert td_independent.dist().batch_shape == expected_batch_shape


class TestTensorIndependentTensorContainerIntegration:
    @pytest.mark.parametrize(
        "param_shape, reinterpreted_batch_ndims",
        [
            ((5,), 0),
            ((5,), 1),
            ((3, 5), 0),
            ((3, 5), 1),
            ((3, 5), 2),
            ((2, 4, 5), 0),
            ((2, 4, 5), 1),
            ((2, 4, 5), 2),
            ((2, 4, 5), 3),
        ],
    )
    def test_compile_compatibility(self, param_shape, reinterpreted_batch_ndims):
        """Core operations should be compatible with torch.compile."""
        loc = torch.randn(*param_shape)
        scale = torch.rand(*param_shape).exp()
        base_distribution = TensorNormal(loc=loc, scale=scale)
        td_independent = TensorIndependent(
            base_distribution=base_distribution,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )

        sample = td_independent.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_independent, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_independent, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_independent, sample, fullgraph=False)


class TestTensorIndependentAPIMatch:
    """
    Tests that the TensorIndependent API matches the PyTorch Independent API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorIndependent matches
        torch.distributions.Independent.
        """
        assert_init_signatures_match(TensorIndependent, Independent)

    def test_properties_match(self):
        """
        Tests that the properties of TensorIndependent match
        torch.distributions.Independent.
        """
        assert_properties_signatures_match(TensorIndependent, Independent)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorIndependent match
        torch.distributions.Independent.
        """
        loc = torch.randn(3, 5)
        scale = torch.rand(3, 5).exp()
        base_distribution = TensorNormal(loc=loc, scale=scale)
        td_independent = TensorIndependent(
            base_distribution=base_distribution, reinterpreted_batch_ndims=1
        )
        assert_property_values_match(td_independent)
