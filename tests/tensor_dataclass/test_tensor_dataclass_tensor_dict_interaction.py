from typing import Tuple

import pytest
import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tensorcontainer.tensor_dict import TensorDict

# Define compile_kwargs locally as it's not exported from compile_utils
compile_kwargs = {"fullgraph": True}


class TensorDataClassWithTensorDict(TensorDataClass):
    """
    A TensorDataClass that contains a TensorDict as a child attribute.
    Used for testing the interaction between TensorDataClass and TensorDict.
    """

    nested_td: TensorDict
    other_tensor: torch.Tensor


def _make_tdc_with_td(
    batch_size: Tuple[int, ...], device: torch.device, requires_grad: bool = False
) -> TensorDataClassWithTensorDict:
    """Helper function to create an instance of TensorDataClassWithTensorDict."""
    # Create a simple, non-nested TensorDict for testing
    nested_td = TensorDict(
        {
            "a": torch.randn(
                *batch_size, 3, device=device, requires_grad=requires_grad
            ),
            "b": torch.randn(
                *batch_size, 4, device=device, requires_grad=requires_grad
            ),
        },
        shape=batch_size,
        device=device,
    )
    other_tensor = torch.randn(
        *batch_size, 5, device=device, requires_grad=requires_grad
    )
    return TensorDataClassWithTensorDict(
        nested_td=nested_td, other_tensor=other_tensor, shape=batch_size, device=device
    )


def _assert_td_close(td1, td2):
    """Recursively compares two TensorDict objects."""
    assert isinstance(td1, TensorDict)
    assert isinstance(td2, TensorDict)
    assert td1.shape == td2.shape
    if td1.device and td2.device:
        assert td1.device.type == td2.device.type
    assert set(td1.keys()) == set(td2.keys())

    for key in td1.keys():
        v1 = td1[key]
        v2 = td2[key]
        if isinstance(v1, TensorDict):
            _assert_td_close(v1, v2)
        else:
            torch.testing.assert_close(v1, v2)


def _assert_tdc_with_td_close(tdc1, tdc2):
    """Compares two TensorDataClassWithTensorDict objects."""
    assert isinstance(tdc1, TensorDataClassWithTensorDict)
    assert isinstance(tdc2, TensorDataClassWithTensorDict)
    assert tdc1.shape == tdc2.shape
    if tdc1.device and tdc2.device:
        assert tdc1.device.type == tdc2.device.type

    _assert_td_close(tdc1.nested_td, tdc2.nested_td)
    torch.testing.assert_close(tdc1.other_tensor, tdc2.other_tensor)


def _run_and_verify_tdc_operation(
    op, tdc, *args, verification_fn=None, skip_compile=False
):
    """
    Helper to run an operation in eager and compiled mode and verify results.
    Optionally performs additional verification steps.
    """
    result_eager = op(tdc, *args)

    if not skip_compile:
        result_compiled = torch.compile(op, fullgraph=compile_kwargs["fullgraph"])(
            tdc, *args
        )
        assert isinstance(result_eager, TensorDataClassWithTensorDict)
        assert isinstance(result_compiled, TensorDataClassWithTensorDict)
        _assert_tdc_with_td_close(result_eager, result_compiled)
    else:
        # If compile is skipped, ensure eager result is still a TensorDataClassWithTensorDict
        assert isinstance(result_eager, TensorDataClassWithTensorDict)

    if verification_fn:
        verification_fn(result_eager, tdc, *args)

    return result_eager


def _getitem_op(tdc_instance, index):
    """Helper function for __getitem__ operation."""
    return tdc_instance[index]


def _verify_getitem_result(result_tdc, original_tdc, index):
    """Verifies the result of a __getitem__ operation."""
    expected_nested_td = original_tdc.nested_td[index]
    expected_other_tensor = original_tdc.other_tensor[index]

    _assert_td_close(result_tdc.nested_td, expected_nested_td)
    torch.testing.assert_close(result_tdc.other_tensor, expected_other_tensor)

    assert result_tdc.shape == expected_nested_td.shape
    assert result_tdc.device == expected_nested_td.device


def _setitem_op(tdc_instance, index, value):
    """Helper function for __setitem__ operation."""
    tdc_instance[index] = value
    return tdc_instance


def _verify_setitem_result(result_tdc, original_tdc_before_op, index, value_to_assign):
    """Verifies the result of a __setitem__ operation."""
    # Create a deepcopy of the original tdc to simulate the expected state after assignment
    expected_tdc = original_tdc_before_op.clone()
    expected_tdc[index] = value_to_assign

    _assert_tdc_with_td_close(result_tdc, expected_tdc)


class TestGetItem:
    @pytest.mark.parametrize(
        "batch_size, idx",
        [
            ((1,), 0),
            ((1,), slice(0, 1)),
            ((2, 3), 0),
            ((2, 3), torch.tensor([0, 1])),
            ((2, 3), slice(0, 1)),
        ],
    )
    def test_getitem(self, batch_size, idx, device):
        """
        Tests __getitem__ operation on a TensorDataClass with a nested TensorDict.

        This test ensures that indexing a TensorDataClass containing a TensorDict
        correctly applies the indexing to both the TensorDict child and other
        tensor attributes.

        Example:
            >>> tdc = _make_tdc_with_td((2,), "cpu")
            >>> tdc[0]
        """
        tdc = _make_tdc_with_td(batch_size, device)

        # Determine if compilation should be skipped for this index type
        # This is still necessary as of PyTorch 2.1.0 for dynamic tensor indexing
        # which can cause graph breaks or recompile issues with torch.compile.
        skip_compile = isinstance(idx, torch.Tensor)

        _run_and_verify_tdc_operation(
            _getitem_op,
            tdc,
            idx,
            verification_fn=lambda result_eager,
            original_tdc,
            *op_args: _verify_getitem_result(result_eager, original_tdc, op_args[0]),
            skip_compile=skip_compile,
        )


class TestDetach:
    @pytest.mark.parametrize("batch_size", [(1,), (2, 3)])
    def test_detach(self, batch_size, device):
        """
        Tests detach operation on a TensorDataClass with a nested TensorDict.

        This test verifies that calling `detach()` on the TensorDataClass
        correctly detaches all tensors, including those within the nested
        TensorDict.
        """
        tdc = _make_tdc_with_td(batch_size, device, requires_grad=True)

        # Verify that requires_grad is initially True for all tensors
        assert tdc.nested_td["a"].requires_grad  # type: ignore
        assert tdc.nested_td["b"].requires_grad  # type: ignore
        assert tdc.other_tensor.requires_grad

        _run_and_verify_tdc_operation(
            self._get_detached_tdc, tdc, verification_fn=self._assert_detach_behavior
        )

    def _get_detached_tdc(self, tdc_instance):
        """Helper method to perform the detach operation."""
        return tdc_instance.detach()

    def _assert_detach_behavior(self, result_tdc, original_tdc, *op_args):
        """
        Verifies the behavior of the detach operation.

        Checks that:
        1. Tensors in the detached result have requires_grad=False.
        2. Tensors in the original TensorDataClass still have requires_grad=True (no in-place modification).
        3. The values of the detached tensors are identical to the original.
        """
        # Verify that the detached tensors in the result have requires_grad=False
        assert not result_tdc.nested_td["a"].requires_grad
        assert not result_tdc.nested_td["b"].requires_grad
        assert not result_tdc.other_tensor.requires_grad

        # Verify that the original tensors in original_tdc still have requires_grad=True
        # This ensures detach creates a new tensor and doesn't modify in-place
        assert original_tdc.nested_td["a"].requires_grad
        assert original_tdc.nested_td["b"].requires_grad
        assert original_tdc.other_tensor.requires_grad

        # Verify that the values of the detached tensors are identical to the original
        # We compare the result with a detached version of the original to ignore requires_grad differences
        _assert_tdc_with_td_close(result_tdc, original_tdc.detach())


class TestView:
    @pytest.mark.parametrize("batch_size", [(2, 4)])
    @pytest.mark.parametrize("new_shape", [(8,), (2, 2, 2)])
    def test_view(self, batch_size, new_shape, device):
        """
        Tests view operation on a TensorDataClass with a nested TensorDict.

        This test checks that the `view()` operation is successfully applied to
        both the nested TensorDict and other tensor attributes of the
        TensorDataClass.
        """
        tdc = _make_tdc_with_td(batch_size, device)
        _run_and_verify_tdc_operation(
            self._perform_view_operation,
            tdc,
            new_shape,
            verification_fn=self._verify_view_result,
        )

    def _perform_view_operation(self, tdc_instance, shape):
        """Helper method to perform the view operation."""
        return tdc_instance.view(*shape)

    def _verify_view_result(self, result_tdc, original_tdc, *op_args):
        """
        Verifies the result of a view operation.

        Checks that:
        1. The nested TensorDict in the result has the expected shape after view.
        2. The other tensor attribute in the result has the expected shape after view.
        3. The values of the viewed tensors are identical to the expected viewed tensors.
        """
        shape = op_args[0]
        expected_nested_td = original_tdc.nested_td.view(*shape)
        event_shape = original_tdc.other_tensor.shape[len(original_tdc.shape) :]
        expected_other_tensor = original_tdc.other_tensor.view(*shape, *event_shape)

        _assert_td_close(result_tdc.nested_td, expected_nested_td)
        torch.testing.assert_close(result_tdc.other_tensor, expected_other_tensor)

        assert result_tdc.shape == expected_nested_td.shape
        assert result_tdc.device == expected_nested_td.device


class TestSetItem:
    @pytest.mark.parametrize(
        "batch_size, idx",
        [
            ((2,), 0),
            ((2, 3), 1),
            ((2, 3), slice(0, 1)),
            ((2, 3), torch.tensor([0, 1])),
        ],
    )
    def test_setitem_success(self, batch_size, idx, device):
        """
        Tests successful __setitem__ operation on a TensorDataClass with a nested TensorDict.

        This test ensures that assigning a TensorDataClass to an indexed
        TensorDataClass correctly updates the data in both the nested TensorDict
        and other tensor attributes, and that eager and compiled executions
        produce the same result.
        """
        dest_tdc = _make_tdc_with_td(batch_size, device)
        # Create a source tdc with a batch size that matches the indexed slice/element
        if isinstance(idx, int):
            source_batch_size = ()
        elif isinstance(idx, slice):
            # For slices, the source batch size should match the size of the slice
            start, stop, step = idx.indices(batch_size[0])
            source_batch_size = (len(range(start, stop, step)),) + batch_size[1:]
        elif isinstance(idx, torch.Tensor):
            source_batch_size = (idx.numel(),) + batch_size[1:]
        else:
            raise NotImplementedError(f"Unsupported index type: {type(idx)}")

        source_tdc = _make_tdc_with_td(source_batch_size, device)

        # Determine if compilation should be skipped for this index type
        skip_compile = isinstance(idx, torch.Tensor)

        _run_and_verify_tdc_operation(
            _setitem_op,
            dest_tdc,
            idx,
            source_tdc,
            verification_fn=lambda result_eager,
            original_tdc,
            *op_args: _verify_setitem_result(
                result_eager, original_tdc, op_args[0], op_args[1]
            ),
            skip_compile=skip_compile,
        )
