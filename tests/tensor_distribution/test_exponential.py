import pytest
import torch
from torch.distributions import Exponential as TorchExponential
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.exponential import TensorExponential
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)


class TestTensorExponentialInitialization:
    @pytest.mark.parametrize(
        "rate_val, expected_batch_shape",
        [
            (1.0, ()),
            (torch.tensor(1.0), ()),
            (torch.tensor([1.0, 2.0]), (2,)),
            (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), (2, 2)),
        ],
    )
    def test_valid_initialization(self, rate_val, expected_batch_shape):
        """Tests that TensorExponential can be instantiated with valid parameters."""
        dist = TensorExponential(rate=rate_val)
        assert isinstance(dist, TensorExponential)
        assert dist.batch_shape == expected_batch_shape
        assert_close(dist.rate, torch.as_tensor(rate_val, dtype=torch.float32))

    @pytest.mark.parametrize(
        "rate_val",
        [
            torch.tensor([-0.1]),  # Invalid rate (non-positive)
            torch.tensor([0.0]),  # Invalid rate (non-positive)
        ],
    )
    def test_invalid_parameter_values_raises_error(
        self, rate_val, with_distributions_validation
    ):
        """Test that invalid rate values raise an error when validation is enabled."""
        with pytest.raises(ValueError, match="Expected parameter rate"):
            TensorExponential(rate=rate_val)


class TestTensorExponentialReferenceComparison:
    @pytest.mark.parametrize(
        "rate_val",
        [
            1.0,
            torch.tensor(1.0),
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ],
    )
    def test_dist_property_and_compilation(self, rate_val):
        """Tests the .dist() property and its compatibility with torch.compile."""
        td_dist = TensorExponential(rate=rate_val)

        # Test .dist() property
        torch_dist = td_dist.dist()
        assert isinstance(torch_dist, TorchExponential)
        assert_close(torch_dist.rate, td_dist.rate)
        assert torch_dist.batch_shape == td_dist.batch_shape
        assert torch_dist.event_shape == td_dist.event_shape

        # Test compilation of .dist()
        def get_dist_rate(td):
            return td.dist().rate

        compiled_rate, _ = run_and_compare_compiled(
            get_dist_rate, td_dist, fullgraph=False
        )
        assert_close(compiled_rate, td_dist.rate)

        def get_dist_batch_shape(td):
            return td.dist().batch_shape

        compiled_batch_shape, _ = run_and_compare_compiled(
            get_dist_batch_shape, td_dist, fullgraph=False
        )
        assert compiled_batch_shape == td_dist.batch_shape

        def get_dist_event_shape(td):
            return td.dist().event_shape

        compiled_event_shape, _ = run_and_compare_compiled(
            get_dist_event_shape, td_dist, fullgraph=False
        )
        assert compiled_event_shape == td_dist.event_shape

    @pytest.mark.parametrize(
        "rate_val",
        [
            1.0,
            torch.tensor(1.0),
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ],
    )
    def test_log_prob_matches_torch_distribution(self, rate_val):
        """Tests that log_prob matches the underlying torch distribution."""
        td_dist = TensorExponential(rate=rate_val)
        value = td_dist.sample() + 0.1  # Ensure value is within support (positive)
        assert_close(td_dist.log_prob(value), td_dist.dist().log_prob(value))

    @pytest.mark.parametrize(
        "rate_val",
        [
            1.0,
            torch.tensor(1.0),
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ],
    )
    def test_log_prob_compilation(self, rate_val):
        """Tests log_prob compatibility with torch.compile."""
        td_dist = TensorExponential(rate=rate_val)
        value = td_dist.sample() + 0.1

        def log_prob_fn(dist, val):
            return dist.log_prob(val)

        eager_log_prob, compiled_log_prob = run_and_compare_compiled(
            log_prob_fn, td_dist, value, fullgraph=False
        )
        assert_close(eager_log_prob, compiled_log_prob)


class TestTensorExponentialAPIMatch:
    def test_init_signatures_match(self):
        """
        Tests that the __init__ signature of TensorExponential matches
        torch.distributions.Exponential.
        """
        assert_init_signatures_match(TensorExponential, TorchExponential)

    def test_properties_match(self):
        """
        Tests that the properties of TensorExponential match
        torch.distributions.Exponential.
        """
        assert_properties_signatures_match(TensorExponential, TorchExponential)

    @pytest.mark.parametrize(
        "rate_val",
        [
            1.0,
            torch.tensor(1.0),
            torch.tensor([1.0, 2.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        ],
    )
    def test_property_values_match(self, rate_val):
        """
        Tests that the property values of TensorExponential match
        torch.distributions.Exponential.
        """
        td_exp = TensorExponential(rate=rate_val)
        assert_property_values_match(td_exp)
