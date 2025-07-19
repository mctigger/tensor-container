"""
Tests for TensorDataClass clone functionality.

This module contains test classes that verify the clone() method behavior
for TensorDataClass instances, including basic cloning, nested structures,
device handling, and compilation compatibility.
"""

import pytest
import torch

from tensorcontainer.tensor_dataclass import TensorDataClass
from tests.conftest import skipif_no_compile, skipif_no_cuda
from tests.tensor_dataclass.conftest import (
    CloneTestClass,
    assert_tensor_equal_and_different_objects,
    create_nested_tensor_dataclass,
)


class TestClone:
    """
    Tests the clone() method of TensorDataClass instances.

    This suite verifies that:
    - Clone creates new instances with independent tensor copies
    - Tensor data values are preserved during cloning
    - Requires_grad flags are maintained in cloned tensors
    - Non-tensor metadata is properly handled during cloning
    - Cloned instances are independent from original instances
    - Nested TensorDataClass structures are properly cloned
    - Empty TensorDataClass instances can be cloned
    - CUDA tensors are correctly cloned with device preservation
    - Clone operations are compatible with torch.compile
    - Mutable metadata follows shallow copy semantics
    """

    def test_clone_creates_new_instance(self, clone_test_instance):
        """
        Clone should create a new TensorDataClass instance.
        The cloned instance should be a different object from the original.
        """
        cloned_td = clone_test_instance.clone()
        assert cloned_td is not clone_test_instance

    def test_clone_creates_new_tensor_objects(self, clone_test_instance):
        """
        Clone should create new tensor objects for all tensor fields.
        The tensor objects should be different from the original tensors.
        """
        cloned_td = clone_test_instance.clone()
        assert cloned_td.a is not clone_test_instance.a
        assert cloned_td.b is not clone_test_instance.b

    def test_clone_preserves_tensor_data(self, clone_test_instance):
        """
        Clone should preserve the data values of all tensors.
        The cloned tensors should have equal values to the original tensors.
        """
        cloned_td = clone_test_instance.clone()
        assert_tensor_equal_and_different_objects(cloned_td.a, clone_test_instance.a)
        assert_tensor_equal_and_different_objects(cloned_td.b, clone_test_instance.b)

    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_clone_preserves_requires_grad(self, clone_test_instance, requires_grad):
        """
        Clone should preserve the requires_grad flag of tensors.
        The cloned tensors should have the same requires_grad setting as originals.
        """
        # Set requires_grad to the test value
        clone_test_instance.a.requires_grad_(requires_grad)
        clone_test_instance.b.requires_grad_(requires_grad)

        cloned_td = clone_test_instance.clone()

        assert cloned_td.a.requires_grad == requires_grad
        assert cloned_td.b.requires_grad == requires_grad

    def test_clone_preserves_non_tensor_metadata(self, clone_test_instance):
        """
        Clone should preserve non-tensor metadata fields.
        The cloned instance should have equal metadata values.
        """
        cloned_td = clone_test_instance.clone()
        assert cloned_td.meta == clone_test_instance.meta

    def test_clone_creates_independent_instance(self, clone_test_instance):
        """
        Clone should create an independent instance.
        Modifying the original should not affect the cloned instance.
        """
        cloned_td = clone_test_instance.clone()
        original_a = clone_test_instance.a.clone()

        # Modify the original
        clone_test_instance.a = torch.zeros_like(clone_test_instance.a)

        # Cloned instance should be unchanged
        assert torch.equal(cloned_td.a, original_a)
        assert not torch.equal(cloned_td.a, clone_test_instance.a)

    def test_clone_nested_structures(self):
        """
        Clone should properly handle nested TensorDataClass structures.
        All nested objects should be cloned independently with preserved data.
        """
        outer, inner = create_nested_tensor_dataclass()
        cloned_outer = outer.clone()

        # Check that all objects are different
        assert cloned_outer is not outer
        assert cloned_outer.inner is not outer.inner  # type: ignore
        assert cloned_outer.inner.c is not outer.inner.c  # type: ignore

        # Check that data is equal but objects are different
        assert_tensor_equal_and_different_objects(cloned_outer.inner.c, outer.inner.c)  # type: ignore

        # Check that requires_grad is preserved
        assert cloned_outer.inner.c.requires_grad == outer.inner.c.requires_grad  # type: ignore

    @skipif_no_cuda
    def test_clone_preserves_cuda_device(self):
        """
        Clone should preserve CUDA device placement for all tensors.
        The cloned tensors should remain on the same CUDA device as originals.
        """
        td = CloneTestClass(
            a=torch.randn(2, 3, device="cuda", requires_grad=True),
            b=torch.ones(2, 3, device="cuda", requires_grad=True),
            shape=torch.Size((2, 3)),
            device=torch.device("cuda"),
        )
        cloned_td = td.clone()

        assert cloned_td.a.device.type == "cuda"  # type: ignore
        assert cloned_td.b.device.type == "cuda"  # type: ignore
        assert cloned_td.device.type == "cuda"  # type: ignore
        assert_tensor_equal_and_different_objects(cloned_td.a, td.a)  # type: ignore
        assert_tensor_equal_and_different_objects(cloned_td.b, td.b)  # type: ignore
        assert cloned_td.a.requires_grad == td.a.requires_grad  # type: ignore
        assert cloned_td.b.requires_grad == td.b.requires_grad  # type: ignore

    @skipif_no_compile
    def test_clone_compilation_compatibility(self):
        """
        Clone should be compatible with torch.compile.
        Functions using clone() should compile and produce consistent results.
        """
        from tests.compile_utils import run_and_compare_compiled

        class MyData(TensorDataClass):
            x: torch.Tensor
            y: torch.Tensor

        def func(td: MyData) -> MyData:
            return td.clone()  # type: ignore

        data = MyData(
            x=torch.ones(3, 4),
            y=torch.zeros(3, 4),
            shape=torch.Size((3, 4)),
            device=torch.device("cpu"),
        )
        run_and_compare_compiled(func, data)

    def test_clone_mutable_metadata_shallow_copy(self):
        """
        Clone should use shallow copy semantics for mutable metadata.
        Mutable metadata objects should be shared between original and clone.
        """

        class MutableMetadata(TensorDataClass):
            a: torch.Tensor
            metadata: list

        td = MutableMetadata(
            shape=torch.Size((2,)),
            device=torch.device("cpu"),
            a=torch.randn(2),
            metadata=[1, 2],
        )

        cloned_td = td.clone()

        # Check that metadata values are equal
        assert cloned_td.metadata == [1, 2]  # type: ignore

        # Check that metadata objects are the same (shallow copy)
        assert cloned_td.metadata is td.metadata  # type: ignore

        # Modify cloned metadata and verify both are affected (shared reference)
        cloned_td.metadata.append(3)  # type: ignore
        assert td.metadata == [1, 2, 3]  # type: ignore
        assert cloned_td.metadata == [1, 2, 3]  # type: ignore
