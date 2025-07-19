import pytest
import torch
from torch._dynamo import exc as dynamo_exc

from tests.compile_utils import run_and_compare_compiled
from tests.conftest import skipif_no_compile
from tests.tensor_dataclass.conftest import (
    compute_cat_shape,
)


class TestCat:
    """Test suite for torch.cat operations on TensorDataclass instances."""

    @staticmethod
    def _cat_operation(tensor_dataclass_list, dim_arg):
        """Helper method for cat operations."""
        return torch.cat(tensor_dataclass_list, dim=dim_arg)

    def _create_test_pair(self, nested_tensor_data_class):
        """
        Creates a pair of dataclass instances, fixing potential inconsistencies
        that can cause pytree errors.
        """
        td1 = nested_tensor_data_class
        td2 = td1.clone()

        # Normalize to prevent pytree context mismatches
        td2.device = td1.device
        td2.shape = td1.shape
        td2.tensor_data_class.device = td1.tensor_data_class.device
        td2.tensor_data_class.shape = td1.tensor_data_class.shape

        return td1, td2

    def _verify_cat_result(self, cat_td, td1, td2, dim):
        """Helper method to verify cat operation results for NestedTensorDataClass."""
        original_batch_shape = td1.shape
        expected_batch_shape = compute_cat_shape(original_batch_shape, dim)
        event_shape = td1.tensor.shape[len(original_batch_shape) :]
        expected_tensor_shape = expected_batch_shape + event_shape

        # The container shape should match the new batch shape
        assert cat_td.shape == expected_batch_shape
        # The tensor shapes should match the new batch shape concatenated with the event shape
        assert cat_td.tensor.shape == expected_tensor_shape
        assert cat_td.tensor_data_class.tensor.shape == expected_tensor_shape

        # Verify metadata is preserved
        assert cat_td.meta_data == td1.meta_data

        # Slicing verification
        actual_dim = dim if dim >= 0 else dim + len(original_batch_shape)
        split_point = original_batch_shape[actual_dim]
        slices = [slice(None)] * len(expected_tensor_shape)
        slices[actual_dim] = slice(0, split_point)
        td1_slice = tuple(slices)
        slices[actual_dim] = slice(split_point, None)
        td2_slice = tuple(slices)

        assert torch.equal(cat_td.tensor[td1_slice], td1.tensor)
        assert torch.equal(cat_td.tensor[td2_slice], td2.tensor)

    @pytest.mark.parametrize("dim", [0, 1, -1, -2])  # Valid batch dimensions
    def test_cat_valid(self, nested_tensor_data_class, dim):
        """Test cat operation with valid batch dimensions."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        cat_td = self._cat_operation([td1, td2], dim)
        self._verify_cat_result(cat_td, td1, td2, dim)

    @pytest.mark.parametrize("dim", [2, 3, -3, -4])  # Invalid event dimensions
    def test_cat_invalid_dim_raises(self, nested_tensor_data_class, dim):
        """Test cat operation raises with invalid event dimensions."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        with pytest.raises(IndexError, match="Dimension out of range"):
            r = self._cat_operation([td1, td2], dim)
            print(r.shape, r.tensor.shape)

    def test_cat_inconsistent_meta_data_raises(self, nested_tensor_data_class):
        """Test cat operation raises with inconsistent metadata."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        td2.meta_data = "different_meta_data"
        with pytest.raises(ValueError, match="Node context mismatch"):
            self._cat_operation([td1, td2], 0)

    def test_cat_inconsistent_shapes_raises(self, nested_tensor_data_class):
        """Test cat operation raises with inconsistent shapes."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)

        # Create an inconsistent shape for one of the tensors.
        inconsistent_shape = list(td1.tensor.shape)
        inconsistent_shape[1] += 1  # Change a non-concatenated batch dimension
        td2.tensor = torch.randn(inconsistent_shape, device=td1.device)

        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            self._cat_operation([td1, td2], 0)

    @skipif_no_compile
    def test_cat_compile(self, nested_tensor_data_class):
        """Test cat operation compilation."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        td2.tensor.mul_(2)
        td2.tensor_data_class.tensor.mul_(2)

        run_and_compare_compiled(self._cat_operation, [td1, td2], 0)

    @skipif_no_compile
    def test_cat_compile_invalid_dim_raises(self, nested_tensor_data_class):
        """Test cat operation raises with invalid dimensions in compile mode."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        compiled_cat_op = torch.compile(self._cat_operation, fullgraph=True)
        # Dynamo detects the IndexError during tracing and raises an Unsupported exception
        with pytest.raises(dynamo_exc.Unsupported) as excinfo:
            compiled_cat_op([td1, td2], 2)  # Invalid event dimension
        assert "IndexError" in str(excinfo.value)

    def test_cat_empty_list_raises(self):
        """Test cat operation raises with empty list."""
        with pytest.raises(RuntimeError, match="expected a non-empty list of Tensors"):
            torch.cat([], dim=0)

    @pytest.mark.parametrize("dim_offset", [2, 3])
    def test_cat_dim_exceeds_batch_ndim(self, nested_tensor_data_class, dim_offset):
        """Test cat operation raises IndexError when dim exceeds batch ndim."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)
        invalid_dim = td1.ndim + dim_offset
        with pytest.raises(IndexError, match="Dimension out of range"):
            self._cat_operation([td1, td2], invalid_dim)

    def test_cat_incompatible_batch_shapes(self, nested_tensor_data_class):
        """Test cat operation raises ValueError when batch shapes are incompatible."""
        td1, td2 = self._create_test_pair(nested_tensor_data_class)

        # Manually set the shape attribute for the test
        td2.shape = (td2.shape[0], td2.shape[1] + 1)

        with pytest.raises(
            ValueError, match="TensorContainer batch shapes must be identical"
        ):
            self._cat_operation([td1, td2], 0)
