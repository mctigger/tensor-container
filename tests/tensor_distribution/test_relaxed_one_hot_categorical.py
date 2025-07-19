import pytest
import torch
from torch.distributions import RelaxedOneHotCategorical as TorchRelaxedCategorical
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

from tensorcontainer.tensor_distribution.relaxed_one_hot_categorical import (
    TensorRelaxedOneHotCategorical,
)
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorRelaxedOneHotCategoricalInitialization:
    @pytest.mark.parametrize(
        "temperature_val, param_type, param_shape, expected_batch_shape",
        [
            (torch.tensor(0.5), "probs", (5,), torch.Size([])),
            (torch.tensor(0.5), "logits", (3, 5), (3,)),
            (torch.tensor(0.5), "probs", (2, 4, 5), (2, 4)),
            (torch.tensor(0.5), "logits", (2, 5), (2,)),
        ],
    )
    def test_valid_initialization(
        self, temperature_val, param_type, param_shape, expected_batch_shape
    ):
        """Test valid initializations with various parameter types and shapes."""
        if param_type == "probs":
            param = torch.rand(*param_shape)
            dist = TensorRelaxedOneHotCategorical(
                temperature=temperature_val, probs=param
            )
        else:
            param = torch.randn(*param_shape)
            dist = TensorRelaxedOneHotCategorical(
                temperature=temperature_val, logits=param
            )

        assert dist.batch_shape == expected_batch_shape
        assert dist.dist().batch_shape == expected_batch_shape

    def test_init_no_params_raises_error(self):
        """A RuntimeError should be raised when neither probs nor logits are provided."""
        with pytest.raises(
            RuntimeError, match="Either 'probs' or 'logits' must be provided."
        ):
            TensorRelaxedOneHotCategorical(temperature=torch.tensor(0.5))


class TestTensorRelaxedOneHotCategoricalTensorContainerIntegration:
    @pytest.mark.parametrize("param_shape", [(5,), (3, 5), (2, 4, 5)])
    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_compile_compatibility(self, param_shape, param_type):
        """Core operations should be compatible with torch.compile."""
        temperature = torch.tensor(0.5, requires_grad=True)
        if param_type == "probs":
            param = torch.rand(*param_shape, requires_grad=True)
            td_dist = TensorRelaxedOneHotCategorical(
                temperature=temperature, probs=param
            )
        else:
            param = torch.randn(*param_shape, requires_grad=True)
            td_dist = TensorRelaxedOneHotCategorical(
                temperature=temperature, logits=param
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

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_copy_method(self, param_type):
        """Test that the .copy() method works correctly."""
        temperature = torch.tensor(0.5)
        if param_type == "probs":
            param = torch.rand(3, 5)
            original_dist = TensorRelaxedOneHotCategorical(
                temperature=temperature, probs=param
            )
        else:
            param = torch.randn(3, 5)
            original_dist = TensorRelaxedOneHotCategorical(
                temperature=temperature, logits=param
            )

        copied_dist = original_dist.copy()

        assert copied_dist is not original_dist
        assert isinstance(copied_dist, TensorRelaxedOneHotCategorical)
        torch.testing.assert_close(original_dist.temperature, copied_dist.temperature)
        if param_type == "probs":
            torch.testing.assert_close(original_dist.probs, copied_dist.probs)
        else:
            torch.testing.assert_close(original_dist.logits, copied_dist.logits)


class TestTensorRelaxedOneHotCategoricalAPIMatch:
    """
    Tests that the TensorRelaxedOneHotCategorical API matches the PyTorch RelaxedOneHotCategorical API.
    """

    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorRelaxedOneHotCategorical matches
        torch.distributions.RelaxedOneHotCategorical.
        """
        assert_init_signatures_match(
            TensorRelaxedOneHotCategorical, TorchRelaxedCategorical
        )

    def test_properties_match(self):
        """
        Tests that the properties of TensorRelaxedOneHotCategorical match
        torch.distributions.RelaxedOneHotCategorical.
        """
        assert_properties_signatures_match(
            TensorRelaxedOneHotCategorical, TorchRelaxedCategorical
        )

    @pytest.mark.parametrize("param_type", ["probs", "logits"])
    def test_property_values_match(self, param_type):
        """
        Tests that the property values of TensorRelaxedOneHotCategorical match
        torch.distributions.RelaxedOneHotCategorical.
        """
        temperature = torch.tensor(0.5)
        if param_type == "probs":
            param = torch.rand(3, 5)
            td_dist = TensorRelaxedOneHotCategorical(
                temperature=temperature, probs=param
            )
        else:
            param = torch.randn(3, 5)
            td_dist = TensorRelaxedOneHotCategorical(
                temperature=temperature, logits=param
            )

        assert_property_values_match(td_dist)


class TestOneHotRelaxedCategoricalBug:
    def test_log_prob(self):
        """
        There is a bug in RelaxedOneHotCategorical https://github.com/pytorch/pytorch/issues/37162
        This test will fail if this bug is fixed in the future. Then it is time to refactor
        TensorRelaxedOneHotCategorical!
        """
        dist = ExpRelaxedCategorical(
            temperature=torch.ones(5, 1), logits=torch.zeros(5, 2)
        )
        assert dist.log_prob(torch.ones(5, 2) / 2).shape == (5, 5)
