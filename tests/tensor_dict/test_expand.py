import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled


class TestExpand:
    """Test the expand method of the TensorDict class."""

    def test_expand_compatible_shape(self):
        """Test that expand with a compatible shape returns a new TensorDict with the expanded shape."""
        td = TensorDict(
            {
                "a": torch.randn(3, 1, 4),
                "b": TensorDict({"c": torch.randn(3, 1, 4)}, shape=(3, 1, 4)),
            },
            shape=(3, 1, 4),
        )

        def expand_fn(t):
            return t.expand(3, 2, 4)

        eager_result, compiled_result = run_and_compare_compiled(expand_fn, td)

        # Check attributes
        assert eager_result.shape == torch.Size([3, 2, 4])
        assert eager_result["a"].shape == torch.Size([3, 2, 4])
        assert eager_result["b"].shape == torch.Size([3, 2, 4])
        assert eager_result["b"]["c"].shape == torch.Size([3, 2, 4])

    def test_expand_is_view(self):
        """Test that the tensors in the new TensorDict are views of the original tensors."""
        td = TensorDict(
            {"a": torch.randn(1, 4)},
            shape=(1, 4),
        )
        original_a = td["a"]
        assert isinstance(original_a, torch.Tensor)
        td_expanded = td.expand(3, 4)

        # Modify original tensor and check if expanded tensor is also modified
        original_a[0, 0] = 100.0
        expanded_a = td_expanded["a"]
        assert isinstance(expanded_a, torch.Tensor)
        assert expanded_a[0, 0] == 100.0
        assert expanded_a[1, 0] == 100.0
        assert expanded_a[2, 0] == 100.0

    def test_expand_dim_one(self):
        """Test that expand correctly handles expanding dimensions of size 1."""
        td = TensorDict(
            {"a": torch.ones(1, 4)},
            shape=(1, 4),
        )

        def expand_fn(t):
            return t.expand(5, 4)

        eager_result, compiled_result = run_and_compare_compiled(expand_fn, td)

        assert eager_result.shape == torch.Size([5, 4])
        assert eager_result["a"].shape == torch.Size([5, 4])
        assert torch.all(eager_result["a"] == 1.0)

    def test_expand_new_dim(self):
        """Test that expand correctly handles adding new dimensions."""
        td = TensorDict(
            {"a": torch.randn(3)},
            shape=(3,),
        )

        def expand_fn(t):
            return t.expand(4, 2, 3)

        eager_result, compiled_result = run_and_compare_compiled(expand_fn, td)

        assert eager_result.shape == torch.Size([4, 2, 3])
        assert eager_result["a"].shape == torch.Size([4, 2, 3])

    def test_expand_incompatible_shape(self):
        """Test that expand raises a RuntimeError when expanding a dimension that is not of size 1 to a different size."""
        td = TensorDict(
            {"a": torch.randn(2, 3)},
            shape=(2, 3),
        )

        def expand_fn(t):
            return t.expand(3, 3)

        with pytest.raises(RuntimeError):
            run_and_compare_compiled(expand_fn, td)

    def test_expand_wildcard(self):
        """Test that expand correctly handles -1 in the shape."""
        td = TensorDict(
            {"a": torch.randn(1, 4)},
            shape=(1, 4),
        )

        def expand_fn(t):
            return t.expand(3, -1)

        eager_result, compiled_result = run_and_compare_compiled(expand_fn, td)

        assert eager_result.shape == torch.Size([3, 4])
        assert eager_result["a"].shape == torch.Size([3, 4])
