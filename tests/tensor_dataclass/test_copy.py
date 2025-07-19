import copy

import pytest
import torch

from tests.conftest import skipif_no_compile


@skipif_no_compile
@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_tensor_dataclass_shallow_copy(copy_test_instance, mode):
    """Test shallow copy of TensorDataclass with non-tensor metadata."""

    def shallow_copy_logic(data):
        return copy.copy(data)

    orig = copy_test_instance
    if mode == "compile":
        # Use the __copy__ method directly for torch.compile compatibility
        copied = orig.__copy__()
    else:
        # Use the wrapper for eager mode
        copied = shallow_copy_logic(orig)

    assert orig is not copied
    assert orig.a is copied.a
    assert orig.b is copied.b
    assert orig.metadata is copied.metadata

    orig.metadata.append(4)
    assert copied.metadata == [1, 2, 3, 4]


@skipif_no_compile
@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_tensor_dataclass_deep_copy(copy_test_instance, mode):
    """Test deep copy of TensorDataclass with non-tensor metadata."""

    orig = copy_test_instance

    if mode == "compile":
        # To test if __deepcopy__ itself is torch.compile compatible,
        # we need to compile a function that directly calls it.
        def compiled_deep_copy_func(data_obj):
            # The __deepcopy__ method expects a memo dictionary.
            # copy.deepcopy normally handles this. For a direct call,
            # we initialize it.
            return data_obj.__deepcopy__()

        compiled_fn = torch.compile(compiled_deep_copy_func, fullgraph=True)
        copied = compiled_fn(orig)
    else:
        # Eager mode uses the standard copy.deepcopy, which should pick up
        # our __deepcopy__ method.
        copied = copy.deepcopy(orig)

    assert orig is not copied
    assert orig.a is not copied.a
    assert orig.b is not copied.b
    assert torch.equal(orig.a, copied.a)
    assert torch.equal(orig.b, copied.b)
    assert orig.metadata is not copied.metadata

    orig.metadata.append(4)
    assert copied.metadata == [1, 2, 3]
