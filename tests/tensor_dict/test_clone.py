import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled


@pytest.mark.skipif_no_compile
class TestTensorDictClone:
    @staticmethod
    def _create_tensor_dict(requires_grad=False, device="cpu"):
        return TensorDict(
            {
                "a": torch.randn(3, 4, requires_grad=requires_grad).to(device),
                "b": torch.randn(3, 4, requires_grad=requires_grad).to(device),
            },
            shape=(3,),
        )

    def test_clone_returns_new_tensordict(self):
        td = self._create_tensor_dict()
        cloned_td = td.clone()
        assert isinstance(cloned_td, TensorDict)
        assert cloned_td is not td

    def test_clone_shape_and_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        td = self._create_tensor_dict(device=device)
        cloned_td = td.clone()
        assert cloned_td.shape == td.shape
        assert cloned_td.device == td.device

    def test_cloned_tensors_are_clones(self):
        td = self._create_tensor_dict(requires_grad=True)
        cloned_td = td.clone()
        for key in td.keys():
            original_tensor = td[key]
            cloned_tensor = cloned_td[key]
            assert isinstance(original_tensor, torch.Tensor)
            assert isinstance(cloned_tensor, torch.Tensor)
            assert torch.equal(original_tensor, cloned_tensor)
            assert original_tensor.data_ptr() != cloned_tensor.data_ptr()
            assert original_tensor.requires_grad == cloned_tensor.requires_grad

    def test_modifying_cloned_tensor_does_not_affect_original(self):
        td = self._create_tensor_dict()
        cloned_td = td.clone()
        original_tensor_a = td["a"]
        cloned_tensor_a = cloned_td["a"]
        assert isinstance(original_tensor_a, torch.Tensor)
        assert isinstance(cloned_tensor_a, torch.Tensor)
        original_value = cloned_tensor_a[0, 0].item()
        cloned_tensor_a[0, 0] = 42
        assert original_tensor_a[0, 0].item() == original_value

    def test_clone_compiled(self):
        td = self._create_tensor_dict(requires_grad=True)

        def clone_fn(t):
            return t.clone()

        eager_result, compiled_result = run_and_compare_compiled(clone_fn, td)

        assert isinstance(compiled_result, TensorDict)
        assert compiled_result is not td
        assert compiled_result.shape == td.shape
        assert compiled_result.device == td.device

        for key in td.keys():
            original_tensor = td[key]
            compiled_tensor = compiled_result[key]
            assert isinstance(original_tensor, torch.Tensor)
            assert isinstance(compiled_tensor, torch.Tensor)
            assert torch.equal(original_tensor, compiled_tensor)
            assert original_tensor.data_ptr() != compiled_tensor.data_ptr()
            assert original_tensor.requires_grad == compiled_tensor.requires_grad
