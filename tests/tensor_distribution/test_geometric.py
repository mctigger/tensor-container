import pytest
import torch
from torch.distributions import Geometric

from tensorcontainer.tensor_distribution.geometric import TensorGeometric
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorGeometricInitialization:
    @pytest.mark.parametrize(
        "param_type, param_shape, expected_batch_shape",
        [
            ("probs", (), ()),
            ("probs", (5,), (5,)),
            ("probs", (3, 5), (3, 5)),
            ("probs", (2, 4, 5), (2, 4, 5)),
            ("logits", (), ()),
            ("logits", (5,), (5,)),
            ("logits", (3, 5), (3, 5)),
            ("logits", (2, 4, 5), (2, 4, 5)),
        ],
    )
    def test_broadcasting_shapes(self, param_type, param_shape, expected_batch_shape):
        """Test that batch_shape is correctly determined by broadcasting."""
        if param_type == "probs":
            probs = torch.rand(param_shape)
            logits = None
        else:  # param_type == "logits"
            logits = torch.rand(param_shape)
            probs = None

        td_geometric = TensorGeometric(probs=probs, logits=logits)
        assert td_geometric.batch_shape == expected_batch_shape
        assert td_geometric.dist().batch_shape == expected_batch_shape

    def test_scalar_parameters(self):
        """Test initialization with scalar parameters."""
        probs = torch.tensor(0.5)
        td_geometric = TensorGeometric(probs=probs)
        assert td_geometric.batch_shape == ()
        assert td_geometric.device == probs.device


class TestTensorGeometricTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, param_shape):
        """Core operations should be compatible with torch.compile."""
        probs = torch.rand(*param_shape)
        td_geometric = TensorGeometric(probs=probs)

        sample = td_geometric.sample()

        def sample_fn(td):
            return td.sample()

        def rsample_fn(td):
            return td.rsample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_geometric, fullgraph=False)
        # Geometric does not have rsample
        # run_and_compare_compiled(rsample_fn, td_geometric, fullgraph=False)
        run_and_compare_compiled(log_prob_fn, td_geometric, sample, fullgraph=False)

    def test_pytree_integration(self):
        """
        We use the copy method as a proxy to ensure pytree integration (e.g. unflattening)
        works correctly.
        """
        probs = torch.rand(3, 5)
        original_dist = TensorGeometric(probs=probs)
        copied_dist = original_dist.copy()

        # Assert that it's a new instance
        assert copied_dist is not original_dist
        assert isinstance(copied_dist, TensorGeometric)


class TestTensorGeometricAPIMatch:
    """
    Tests that the TensorGeometric API matches the PyTorch Geometric API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorGeometric matches
        torch.distributions.Geometric.
        """
        assert_init_signatures_match(TensorGeometric, Geometric)

    def test_properties_match(self):
        """
        Tests that the properties of TensorGeometric match
        torch.distributions.Geometric.
        """
        assert_properties_signatures_match(TensorGeometric, Geometric)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorGeometric match
        torch.distributions.Geometric.
        """
        probs = torch.rand(3, 5)
        td_geometric = TensorGeometric(probs=probs)
        assert_property_values_match(td_geometric)
