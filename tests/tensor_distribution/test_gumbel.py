import pytest
import torch
from torch.distributions import Gumbel

from tensorcontainer.tensor_distribution.gumbel import TensorGumbel
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorGumbelInitialization:
    @pytest.mark.parametrize(
        "param_shape, expected_batch_shape",
        [
            ((), ()),
            ((5,), (5,)),
            ((3, 5), (3, 5)),
            ((2, 4, 5), (2, 4, 5)),
        ],
    )
    def test_broadcasting_shapes(self, param_shape, expected_batch_shape):
        """Test that batch_shape is correctly determined by broadcasting."""
        loc = torch.rand(param_shape)
        scale = torch.rand(param_shape) + 0.1  # scale must be positive

        td_gumbel = TensorGumbel(loc=loc, scale=scale)
        assert td_gumbel.batch_shape == expected_batch_shape
        assert td_gumbel.dist().batch_shape == expected_batch_shape

    def test_scalar_parameters(self):
        """Test initialization with scalar parameters."""
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        td_gumbel = TensorGumbel(loc=loc, scale=scale)
        assert td_gumbel.batch_shape == ()
        assert td_gumbel.device == loc.device


class TestTensorGumbelTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        loc = torch.rand(*param_shape)
        scale = torch.rand(*param_shape) + 0.1
        td_gumbel = TensorGumbel(loc=loc, scale=scale)

        sample = td_gumbel.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_gumbel, fullgraph=False)
        run_and_compare_compiled(rsample_fn, td_gumbel, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_gumbel, sample, fullgraph=False)

    def test_pytree_integration(self):
        """
        We use the copy method as a proxy to ensure pytree integration (e.g. unflattening)
        works correctly.
        """
        loc = torch.rand(3, 5)
        scale = torch.rand(3, 5) + 0.1
        original_dist = TensorGumbel(loc=loc, scale=scale)
        copied_dist = original_dist.copy()

        # Assert that it's a new instance
        assert copied_dist is not original_dist
        assert isinstance(copied_dist, TensorGumbel)


class TestTensorGumbelAPIMatch:
    """
    Tests that the TensorGumbel API matches the PyTorch Gumbel API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorGumbel matches
        torch.distributions.Gumbel.
        """
        assert_init_signatures_match(TensorGumbel, Gumbel)

    def test_properties_match(self):
        """
        Tests that the properties of TensorGumbel match
        torch.distributions.Gumbel.
        """
        assert_properties_signatures_match(TensorGumbel, Gumbel)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorGumbel match
        torch.distributions.Gumbel.
        """
        loc = torch.rand(3, 5)
        scale = torch.rand(3, 5) + 0.1
        td_gumbel = TensorGumbel(loc=loc, scale=scale)
        assert_property_values_match(td_gumbel)
