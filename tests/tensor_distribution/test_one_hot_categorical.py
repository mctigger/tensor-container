import pytest
import torch
from torch.distributions import OneHotCategorical

from tensorcontainer.tensor_distribution.one_hot_categorical import (
    TensorOneHotCategorical,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorOneHotCategoricalInitialization:
    def test_init_no_params_raises_error(self):
        """A ValueError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            RuntimeError, match="Either 'probs' or 'logits' must be provided."
        ):
            TensorOneHotCategorical()


class TestTensorOneHotCategoricalTensorContainerIntegration:
    @pytest.mark.parametrize("logits_shape", [(5,), (3, 5), (2, 4, 5)])
    def test_compile_compatibility(self, logits_shape):
        """Core operations should be compatible with torch.compile."""
        logits = torch.randn(*logits_shape, requires_grad=True)
        td_one_hot_categorical = TensorOneHotCategorical(logits=logits)
        sample = td_one_hot_categorical.sample()

        def sample_fn(td):
            return td.sample()

        def log_prob_fn(td, s):
            return td.log_prob(s)

        run_and_compare_compiled(sample_fn, td_one_hot_categorical, fullgraph=False)
        run_and_compare_compiled(
            log_prob_fn, td_one_hot_categorical, sample, fullgraph=False
        )

    def test_pytree_integration(self):
        """
        We use the copy method as a proxy to ensure pytree integration (e.g. unflattening)
        works correctly.
        """
        logits = torch.randn(3, 5)
        original_dist = TensorOneHotCategorical(logits=logits)
        copied_dist = original_dist.copy()

        # Assert that it's a new instance
        assert copied_dist is not original_dist
        assert isinstance(copied_dist, TensorOneHotCategorical)


class TestTensorOneHotCategoricalAPIMatch:
    """
    Tests that the TensorOneHotCategorical API matches the PyTorch OneHotCategorical API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorOneHotCategorical matches
        torch.distributions.OneHotCategorical.
        """
        assert_init_signatures_match(TensorOneHotCategorical, OneHotCategorical)

    def test_properties_match(self):
        """
        Tests that the properties of TensorOneHotCategorical match
        torch.distributions.OneHotCategorical.
        """
        assert_properties_signatures_match(TensorOneHotCategorical, OneHotCategorical)

    def test_property_values_match(self):
        """
        Tests that the property values of TensorOneHotCategorical match
        torch.distributions.OneHotCategorical.
        """
        logits = torch.randn(3, 5)
        td_one_hot_cat = TensorOneHotCategorical(logits=logits)
        assert_property_values_match(td_one_hot_cat)
