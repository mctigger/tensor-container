import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled


class TestTensorDictDetach:
    @staticmethod
    def _create_tensor_dict(requires_grad=False):
        return TensorDict(
            {
                "a": torch.randn(3, 4, requires_grad=requires_grad),
                "b": torch.randn(3, 4, requires_grad=requires_grad),
            },
            shape=(3,),
        )

    def test_detach_returns_new_tensordict(self):
        td = self._create_tensor_dict()
        detached_td = td.detach()
        assert isinstance(detached_td, TensorDict)
        assert detached_td is not td

    def test_detached_tensors_require_grad_false(self):
        td = self._create_tensor_dict(requires_grad=True)
        detached_td = td.detach()
        for key in detached_td.keys():
            assert not detached_td[key].requires_grad

    def test_original_tensordict_unchanged(self):
        td = self._create_tensor_dict(requires_grad=True)
        td.detach()
        for key in td.keys():
            assert td[key].requires_grad

    def test_detached_tensors_share_storage(self):
        td = self._create_tensor_dict(requires_grad=True)
        detached_td = td.detach()
        for key in td.keys():
            assert td[key].data_ptr() == detached_td[key].data_ptr()

    def test_detach_compiled(self):
        td = self._create_tensor_dict(requires_grad=True)

        def detach_fn(t):
            return t.detach()

        eager_result, compiled_result = run_and_compare_compiled(detach_fn, td)

        assert isinstance(compiled_result, TensorDict)
        assert compiled_result is not td
        for key in compiled_result.keys():
            assert not compiled_result[key].requires_grad
        # The original td should be unchanged
        for key in td.keys():
            assert td[key].requires_grad
        # Check for shared storage
        for key in td.keys():
            assert td[key].data_ptr() == compiled_result[key].data_ptr()
