import pytest
import torch

from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import (
    ShapeTestClass,
    assert_shape_consistency,
)

# Note: The view and reshape tests have been moved to test_view_reshape.py
# This file now focuses on shape inference and other shape-related operations


class TestShapeOperations:
    """Test suite for shape-related operations on TensorDataclass instances."""

    def test_shape_inference_on_unflatten(self, shape_test_instance):
        """Test that shape is correctly inferred during pytree unflatten operations."""
        original = shape_test_instance

        # Flatten and modify leaves
        leaves, context = original._pytree_flatten()
        modified_leaves = [t * 2 for t in leaves]

        # Reconstruct with original context
        reconstructed = ShapeTestClass._pytree_unflatten(modified_leaves, context)
        assert_shape_consistency(reconstructed, original.shape)

    def test_invalid_shape_raises(self, shape_test_instance):
        """Test that invalid shape operations raise appropriate errors."""
        td = shape_test_instance

        with pytest.raises(RuntimeError):
            td.view(21)  # Invalid size for (4, 5) -> 20 elements

    @skipif_no_compile
    def test_shape_compile(self, shape_test_instance):
        """Test that shape operations work correctly with torch.compile."""
        td = shape_test_instance

        def view_fn(td_input):
            return td_input.view(20)

        compiled_fn = torch.compile(view_fn, fullgraph=True)
        result = compiled_fn(td)

        assert result.shape == (20,)
        assert result.a.shape == (20,)

    def test_zero_sized_batch(self):
        """Test initialization and operations with a batch size of 0."""
        td = ShapeTestClass(
            shape=(0, 10),
            device=torch.device("cpu"),
            a=torch.randn(0, 10),
            b=torch.randn(0, 10),
        )

        assert_shape_consistency(td, (0, 10))

        # Test clone
        cloned_td = td.clone()
        assert isinstance(cloned_td, ShapeTestClass)
        assert_shape_consistency(cloned_td, (0, 10))
        assert torch.equal(cloned_td.a, td.a)
        assert cloned_td.a is not td.a

        # Test stack
        stacked_td = torch.stack([td, td], dim=0)  # type: ignore
        assert isinstance(stacked_td, ShapeTestClass)
        assert_shape_consistency(stacked_td, (2, 0, 10))

    def test_inconsistent_trailing_shapes(self):
        """Test initialization with tensors that have different trailing shapes."""
        try:
            td = ShapeTestClass(
                shape=(4,),
                device=torch.device("cpu"),
                a=torch.randn(4, 10),
                b=torch.randn(4, 5),  # Different trailing dimension
            )
            assert td.shape == (4,)
            assert td.a.shape == (4, 10)
            assert td.b.shape == (4, 5)
        except ValueError:
            pytest.fail("Initialization failed with inconsistent trailing shapes.")

    @pytest.mark.parametrize(
        "original_shape,expected_elements",
        [
            ((2, 3), 6),
            ((4, 5), 20),
            ((1, 10), 10),
            ((3, 2, 2), 12),
        ],
    )
    def test_shape_numel_consistency(self, original_shape, expected_elements):
        """Test that shape and tensor element count are consistent."""
        td = ShapeTestClass(
            shape=original_shape,
            device=torch.device("cpu"),
            a=torch.randn(original_shape),
            b=torch.ones(original_shape),
        )

        assert td.a.numel() == expected_elements
        assert td.b.numel() == expected_elements

    @skipif_no_compile
    @pytest.mark.parametrize(
        "original_shape,view_shape",
        [
            ((2, 3), (6,)),
            ((4, 5), (20,)),
            ((2, 2, 3), (4, 3)),
        ],
    )
    def test_shape_view_compile(self, original_shape, view_shape):
        """Test that view operations work correctly with torch.compile."""
        from tests.compile_utils import run_and_compare_compiled

        td = ShapeTestClass(
            shape=original_shape,
            device=torch.device("cpu"),
            a=torch.randn(original_shape),
            b=torch.ones(original_shape),
        )

        def view_fn(td_input):
            return td_input.view(*view_shape)

        run_and_compare_compiled(view_fn, td)
