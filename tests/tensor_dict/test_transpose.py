import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled


class TestTranspose:
    @staticmethod
    def _create_tensor_dict(shape, event_shapes, requires_grad=False, device="cpu"):
        data = {
            f"a_{i}": torch.randn(*shape, *event_shape, requires_grad=requires_grad).to(
                device
            )
            for i, event_shape in enumerate(event_shapes)
        }
        return TensorDict(data, shape)

    @pytest.mark.parametrize(
        "dim0, dim1", [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    )
    def test_transpose(self, dim0, dim1):
        shape = (3, 4, 5)
        td = self._create_tensor_dict(shape, event_shapes=[(), (6,)])
        t_td = td.transpose(dim0, dim1)

        # Check shape
        expected_shape = list(td.shape)
        expected_shape[dim0], expected_shape[dim1] = (
            expected_shape[dim1],
            expected_shape[dim0],
        )
        expected_shape = torch.Size(expected_shape)
        assert t_td.shape == expected_shape

        # Check that tensors are views
        assert t_td["a_0"]._is_view()
        assert t_td["a_1"]._is_view()
        assert t_td["a_0"]._base is td["a_0"]
        assert t_td["a_1"]._base is td["a_1"]

        # Check that event dims are untouched
        assert t_td["a_0"].shape == expected_shape
        assert t_td["a_1"].shape == (*expected_shape, 6)

    @pytest.mark.parametrize(
        "dim0, dim1", [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    )
    def test_transpose_compile(self, dim0, dim1):
        shape = (3, 4, 5)
        td = self._create_tensor_dict(shape, event_shapes=[(), (6,)])

        def transpose_fn(t):
            return t.transpose(dim0, dim1)

        eager_result, compiled_result = run_and_compare_compiled(transpose_fn, td)

        # Check shape
        expected_shape = list(td.shape)
        expected_shape[dim0], expected_shape[dim1] = (
            expected_shape[dim1],
            expected_shape[dim0],
        )
        expected_shape = torch.Size(expected_shape)
        assert eager_result.shape == expected_shape
        assert compiled_result.shape == expected_shape

    def test_t(self):
        shape = (3, 4, 5)
        td = self._create_tensor_dict(shape, event_shapes=[(), (6,)])
        t_td = td.t()
        t_td_explicit = td.transpose(0, 1)

        assert t_td.shape == t_td_explicit.shape
        torch.testing.assert_close(t_td["a_0"], t_td_explicit["a_0"])
        torch.testing.assert_close(t_td["a_1"], t_td_explicit["a_1"])

        # Check that tensors are views
        assert t_td["a_0"]._is_view()
        assert t_td["a_1"]._is_view()
        assert t_td["a_0"]._base is td["a_0"]
        assert t_td["a_1"]._base is td["a_1"]

    def test_t_compile(self):
        shape = (3, 4, 5)
        td = self._create_tensor_dict(shape, event_shapes=[(), (6,)])

        def t_fn(t):
            return t.t()

        eager_result, compiled_result = run_and_compare_compiled(t_fn, td)

        # Check shape
        t_td_explicit = td.transpose(0, 1)
        assert eager_result.shape == t_td_explicit.shape
        assert compiled_result.shape == t_td_explicit.shape

    def test_t_error_1d(self):
        td = self._create_tensor_dict((3,), event_shapes=[()])
        with pytest.raises(RuntimeError):
            td.t()
