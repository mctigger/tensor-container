import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled
from tests.conftest import skipif_no_compile


@pytest.fixture
def nested_dict():
    def _make(shape):
        return {
            "x": {
                "a": torch.arange(0, 4).reshape(*shape),
                "b": torch.arange(4, 8).reshape(*shape),
            },
            "y": torch.arange(8, 12).reshape(*shape),
        }

    return _make


def test_copy_returns_distinct_tensordict_but_shares_leaf_tensors(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))
    td_copy = td.copy()

    # top‐level object is new
    assert td_copy is not td
    # shape and device preserved
    assert td_copy.shape == td.shape
    assert td_copy.device == td.device

    for key in td:
        val = td[key]
        copied_val = td_copy[key]
        # nested TensorDicts should be new objects
        if isinstance(val, TensorDict):
            assert isinstance(copied_val, TensorDict)
            assert copied_val is not val
            # their leaves must still be the same tensor objects
            for leaf_key in val:
                assert copied_val[leaf_key] is val[leaf_key]
        else:
            # leaf tensors should be the same object
            assert copied_val is val


@skipif_no_compile
def test_copy_returns_distinct_tensordict_but_shares_leaf_tensors_compiled(nested_dict):
    """Test that copy works correctly with torch.compile."""

    def copy_fn(td):
        return td.copy()

    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))

    eager_result, compiled_result = run_and_compare_compiled(copy_fn, td)

    # Additional checks specific to copy behavior
    # top‐level object is new (for eager result)
    assert eager_result is not td

    for key in td:
        val = td[key]
        copied_val = eager_result[key]
        # nested TensorDicts should be new objects
        if isinstance(val, TensorDict):
            assert isinstance(copied_val, TensorDict)
            assert copied_val is not val
            # their leaves must still be the same tensor objects
            for leaf_key in val:
                assert copied_val[leaf_key] is val[leaf_key]
        else:
            # leaf tensors should be the same object
            assert copied_val is val


def test_mutating_nested_copy_does_not_affect_original(nested_dict):
    data = nested_dict((4, 1))
    td = TensorDict(data, shape=(4, 1))
    td_copy = td.copy()

    # modify the copy's nested structure
    td_copy["x"]["c"] = torch.tensor([[42], [43], [44], [45]])
    assert "c" in td_copy["x"]
    # original remains unchanged
    assert "c" not in td["x"]

    # remove a leaf from the copy
    del td_copy["x"]["a"]
    assert "a" not in td_copy["x"]
    # original still has its leaf
    assert "a" in td["x"]


def test_copy_tensor_vs_container_identity(nested_dict):
    """
    Checks that after a .copy() call, torch.Tensor items are the same
    reference, while TensorContainer items are new objects.
    """
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))
    td_copy = td.copy()

    # The top-level TensorDict is a TensorContainer, so it should be a new object.
    assert td is not td_copy

    # Check items within the TensorDict
    for key, value in td.items():
        copied_value = td_copy[key]
        if isinstance(value, TensorDict):
            assert copied_value is not value, (
                f"TensorContainer at key '{key}' should be a new object"
            )
            # For nested containers, check that their leaf tensors are shared
            for leaf_key, leaf_value in value.items():
                assert copied_value[leaf_key] is leaf_value, (
                    f"Leaf tensor at '{key}.{leaf_key}' should be shared"
                )
        elif isinstance(value, torch.Tensor):
            assert copied_value is value, (
                f"Tensor at key '{key}' should be the same object"
            )


@skipif_no_compile
def test_mutating_nested_copy_does_not_affect_original_compiled(nested_dict):
    """Test that copy works correctly with torch.compile."""

    def copy_fn(td):
        # Just return the copy, we'll mutate it outside the compiled function
        return td.copy()

    data = nested_dict((4, 1))
    td = TensorDict(data, shape=(4, 1))

    eager_copy, compiled_copy = run_and_compare_compiled(copy_fn, td)

    # Now mutate the copies outside the compiled function
    # This tests that the copy operation worked correctly with torch.compile

    # Modify the eager copy
    eager_copy["x"]["c"] = torch.tensor([[42], [43], [44], [45]])
    assert "c" in eager_copy["x"]
    assert "c" not in td["x"]

    # Modify the compiled copy
    compiled_copy["x"]["c"] = torch.tensor([[42], [43], [44], [45]])
    assert "c" in compiled_copy["x"]
    assert "c" not in td["x"]


def test_copy_of_empty_tensor_dict(nested_dict):
    # an empty dict should still copy correctly
    td = TensorDict({}, shape=())
    td_copy = td.copy()
    assert isinstance(td_copy, TensorDict)
    assert td_copy is not td
    assert td_copy.shape == torch.Size([])
    assert len(td_copy) == 0


@skipif_no_compile
def test_copy_of_empty_tensor_dict_compiled():
    """Test that copying an empty TensorDict works with torch.compile."""

    def copy_empty_td(td):
        return td.copy()

    td = TensorDict({}, shape=())

    eager_result, compiled_result = run_and_compare_compiled(copy_empty_td, td)

    # Additional checks specific to empty TensorDict
    assert isinstance(eager_result, TensorDict)
    assert eager_result is not td
    assert eager_result.shape == torch.Size([])
    assert len(eager_result) == 0


def test_copy_with_pytree(nested_dict):
    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))
    td_copy = td.copy()

    # top‐level object is new
    assert td_copy is not td
    # shape and device preserved
    assert td_copy.shape == td.shape
    assert td_copy.device == td.device

    for key in td:
        val = td[key]
        copied_val = td_copy[key]
        # nested TensorDicts should be new objects
        if isinstance(val, TensorDict):
            assert isinstance(copied_val, TensorDict)
            assert copied_val is not val
            # their leaves must still be the same tensor objects
            for leaf_key in val:
                assert copied_val[leaf_key] is val[leaf_key]
        else:
            # leaf tensors should be the same object
            assert copied_val is val


@skipif_no_compile
def test_copy_with_pytree_compiled(nested_dict):
    """Test that copy works correctly with pytree and torch.compile."""

    def copy_fn(td):
        return td.copy()

    data = nested_dict((2, 2))
    td = TensorDict(data, shape=(2, 2))

    eager_result, compiled_result = run_and_compare_compiled(copy_fn, td)

    # Additional checks specific to copy behavior
    # top‐level object is new (for eager result)
    assert eager_result is not td

    for key in td:
        val = td[key]
        copied_val = eager_result[key]
        # nested TensorDicts should be new objects
        if isinstance(val, TensorDict):
            assert isinstance(copied_val, TensorDict)
            assert copied_val is not val
            # their leaves must still be the same tensor objects
            for leaf_key in val:
                assert copied_val[leaf_key] is val[leaf_key]
        else:
            # leaf tensors should be the same object
            assert copied_val is val


@skipif_no_compile
def test_copy_inside_compile():
    """Test that creating and copying a TensorDict inside torch.compile works."""

    def compiled_fn(x):
        td = TensorDict({"a": x, "b": x + 1}, shape=(1,))
        td_copy = td.copy()
        return td_copy

    x = torch.randn(1)
    compiled = torch.compile(compiled_fn, fullgraph=True)
    td_copy = compiled(x)

    assert isinstance(td_copy, TensorDict)
    assert td_copy is not None
    assert torch.equal(td_copy["a"], x)
    assert torch.equal(td_copy["b"], x + 1)
