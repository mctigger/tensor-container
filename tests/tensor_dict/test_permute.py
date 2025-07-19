import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled


class TestPermute:
    @pytest.mark.parametrize(
        "batch_dims, perm",
        [
            ((2, 3), (1, 0)),
            ((2, 3, 4), (2, 0, 1)),
            ((5,), (0,)),
        ],
    )
    def test_permute_shape(self, batch_dims, perm):
        """Test that permute returns a new TensorDict with the permuted shape."""
        td = TensorDict(
            {"a": torch.randn(*batch_dims, 5), "b": torch.randn(*batch_dims, 6, 7)},
            shape=batch_dims,
        )

        def permute_fn(t):
            return t.permute(*perm)

        eager_result, _ = run_and_compare_compiled(permute_fn, td)

        # Check shape
        expected_shape = tuple(batch_dims[i] for i in perm)
        assert eager_result.shape == torch.Size(expected_shape)
        assert eager_result["a"].shape == (*expected_shape, 5)
        assert eager_result["b"].shape == (*expected_shape, 6, 7)

    def test_permute_is_view(self):
        """Test that the tensors in the new TensorDict are views of the original tensors."""
        td = TensorDict(
            {"a": torch.randn(2, 3)},
            shape=(2, 3),
        )
        original_a = td["a"]
        assert isinstance(original_a, torch.Tensor)
        td_permuted = td.permute(1, 0)

        # Modify original tensor and check if permuted tensor is also modified
        original_a[0, 0] = 100.0
        permuted_a = td_permuted["a"]
        assert isinstance(permuted_a, torch.Tensor)
        assert permuted_a[0, 0] == 100.0

    @pytest.mark.parametrize(
        "batch_dims, perm",
        [
            ((2, 3), (0,)),  # Not enough dims
            ((2, 3), (0, 0)),  # Duplicate dims
            ((2, 3), (0, 1, 2)),  # Too many dims
            ((2, 3), (0, 2)),  # Out of bounds
        ],
    )
    def test_permute_invalid_dims(self, batch_dims, perm):
        """Test that permute raises a RuntimeError for invalid permutations."""
        td = TensorDict({"a": torch.randn(*batch_dims)}, shape=batch_dims)

        def permute_fn(t):
            return t.permute(*perm)

        with pytest.raises(RuntimeError):
            permute_fn(td)

        with pytest.raises(Exception):
            # Compiled code might raise a different error type
            run_and_compare_compiled(permute_fn, td)
