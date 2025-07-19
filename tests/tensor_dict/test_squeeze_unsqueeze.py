import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled


@pytest.mark.skipif_no_compile
class TestSqueeze:
    @staticmethod
    def _create_tensor_dict(shape, requires_grad=False, device="cpu"):
        return TensorDict(
            {
                "a": torch.randn(*shape, requires_grad=requires_grad).to(device),
                "b": torch.randn(*shape, requires_grad=requires_grad).to(device),
            },
            shape=shape,
        )

    def test_squeeze_dim(self):
        td = self._create_tensor_dict(shape=(3, 1, 4))

        def squeeze_fn(t):
            return t.squeeze(1)

        eager_result, compiled_result = run_and_compare_compiled(squeeze_fn, td)

        assert eager_result.shape == (3, 4)
        assert compiled_result.shape == (3, 4)

    def test_squeeze_dim_noop(self):
        td = self._create_tensor_dict(shape=(3, 4))

        def squeeze_fn(t):
            return t.squeeze(0)

        eager_result, compiled_result = run_and_compare_compiled(squeeze_fn, td)

        assert eager_result.shape == (3, 4)
        assert compiled_result.shape == (3, 4)

    def test_squeeze_all(self):
        td = self._create_tensor_dict(shape=(1, 3, 1, 4, 1))

        def squeeze_fn(t):
            return t.squeeze()

        eager_result, compiled_result = run_and_compare_compiled(squeeze_fn, td)

        assert eager_result.shape == (3, 4)
        assert compiled_result.shape == (3, 4)


class TestUnsqueeze:
    @staticmethod
    def _create_tensor_dict(shape, requires_grad=False, device="cpu"):
        return TensorDict(
            {
                "a": torch.randn(*shape, requires_grad=requires_grad).to(device),
                "b": torch.randn(*shape, requires_grad=requires_grad).to(device),
            },
            shape=shape,
        )

    def test_unsqueeze(self):
        td = self._create_tensor_dict(shape=(3, 4))

        def unsqueeze_fn(t):
            return t.unsqueeze(1)

        eager_result, compiled_result = run_and_compare_compiled(unsqueeze_fn, td)

        assert eager_result.shape == (3, 1, 4)
        assert compiled_result.shape == (3, 1, 4)
