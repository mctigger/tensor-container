import pytest
import torch
from torch.distributions import Chi2 as TorchChi2
from torch.testing import assert_close

from tensorcontainer.tensor_distribution.chi2 import TensorChi2
from tests.compile_utils import run_and_compare_compiled
from tests.tensor_distribution.conftest import (
    assert_init_signatures_match,
    assert_properties_signatures_match,
    assert_property_values_match,
)

# Define TEST_CASES specific to Chi2's parameters and shapes.
TEST_CASES = [
    # (batch_shape, df_shape)
    ((), (1,)),  # Scalar distribution
    ((3,), (3,)),  # 1D batch shape
    ((2, 4), (2, 4)),  # 2D batch shape
    ((2, 3), (2, 3)),  # Reinterpreted batch dim
    ((2, 3, 4), (2, 3, 4)),  # Multiple reinterpreted batch dims
]


def _generate_params(batch_shape, df_shape, device):
    # Chi2 df must be positive
    df = torch.rand(batch_shape + df_shape, device=device) + 0.1
    return df


class TestTensorChi2:
    """
    Tests that TensorChi2 behaves consistently with torch.distributions.Chi2.
    """

    def test_init_signatures_match(self):
        assert_init_signatures_match(TensorChi2, TorchChi2)

    def test_properties_signatures_match(self):
        assert_properties_signatures_match(TensorChi2, TorchChi2)

    @pytest.mark.parametrize(
        "batch_shape, df_shape",
        TEST_CASES,
        ids=[f"batch={bs},df={dfs}" for bs, dfs in TEST_CASES],
    )
    def test_property_values_match(self, batch_shape, df_shape):
        df_val = _generate_params(batch_shape, df_shape, "cpu")
        td_dist = TensorChi2(df=df_val)
        assert_property_values_match(td_dist)

    @pytest.mark.parametrize(
        "batch_shape, df_shape",
        TEST_CASES,
        ids=[f"batch={bs},df={dfs}" for bs, dfs in TEST_CASES],
    )
    def test_dist_property_and_compilation(self, batch_shape, df_shape):
        """
        Tests the .dist() property and its compatibility with torch.compile.
        """
        df_val = _generate_params(batch_shape, df_shape, "cpu")

        td_dist = TensorChi2(df=df_val)

        # Test .dist() property
        torch_dist = td_dist.dist()
        assert isinstance(torch_dist, TorchChi2)
        assert_close(torch_dist.df, df_val)
        assert torch_dist.batch_shape == td_dist.batch_shape
        assert torch_dist.event_shape == td_dist.event_shape

        # Test compilation of .dist()
        def get_dist_attributes(td):
            dist_instance = td.dist()
            return (
                dist_instance.df,
                dist_instance.batch_shape,
                dist_instance.event_shape,
            )

        eager_attrs, compiled_attrs = run_and_compare_compiled(
            get_dist_attributes, td_dist, fullgraph=False
        )

        # Unpack the attributes
        eager_df, eager_batch_shape, eager_event_shape = eager_attrs
        compiled_df, compiled_batch_shape, compiled_event_shape = compiled_attrs

        assert_close(compiled_df, eager_df)
        assert compiled_batch_shape == eager_batch_shape
        assert compiled_event_shape == eager_event_shape

    @pytest.mark.parametrize(
        "batch_shape, df_shape",
        TEST_CASES,
        ids=[f"batch={bs},df={dfs}" for bs, dfs in TEST_CASES],
    )
    def test_log_prob_matches_torch_distribution(self, batch_shape, df_shape):
        df_val = _generate_params(batch_shape, df_shape, "cpu")
        td_dist = TensorChi2(df=df_val)

        # Generate a value within the support of Chi2 (positive)
        value = td_dist.sample() + 0.1

        assert_close(td_dist.log_prob(value), td_dist.dist().log_prob(value))

    @pytest.mark.parametrize(
        "batch_shape, df_shape",
        TEST_CASES,
        ids=[f"batch={bs},df={dfs}" for bs, dfs in TEST_CASES],
    )
    def test_log_prob_compilation(self, batch_shape, df_shape):
        df_val = _generate_params(batch_shape, df_shape, "cpu")
        td_dist = TensorChi2(df=df_val)
        value = td_dist.sample() + 0.1

        def log_prob_fn(dist, val):
            return dist.log_prob(val)

        eager_log_prob, compiled_log_prob = run_and_compare_compiled(
            log_prob_fn, td_dist, value, fullgraph=False
        )
        assert_close(eager_log_prob, compiled_log_prob)

    @pytest.mark.parametrize(
        "df_val",
        [
            torch.tensor([-0.1]),  # Invalid df (non-positive)
            torch.tensor([0.0]),  # Invalid df (non-positive)
        ],
    )
    def test_invalid_parameter_values_raises_error(self, df_val):
        with pytest.raises(ValueError):
            TensorChi2(df=df_val)
