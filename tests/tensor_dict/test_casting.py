import pytest
import torch
import torch.utils._pytree as pytree

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled

from .test_device import are_devices_equal


@pytest.mark.skipif_no_compile
class TestCasting:
    @pytest.mark.parametrize(
        "method_name, target_dtype",
        [
            ("float", torch.float32),
            ("double", torch.float64),
            ("half", torch.float16),
            ("long", torch.int64),
            ("int", torch.int32),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu"])
    def test_casting_methods(self, method_name, target_dtype, device):
        td = TensorDict(
            {
                "a": torch.ones(3, 4, dtype=torch.float32, device=device),
                "b": torch.zeros(3, 4, 5, dtype=torch.float32, device=device),
            },
            shape=(3, 4),
            device=device,
        )

        def casting_op(t):
            casting_method = getattr(t, method_name)
            return casting_method()

        eager_result, compiled_result = run_and_compare_compiled(casting_op, td)
        new_td = eager_result

        assert isinstance(new_td, TensorDict)
        assert new_td is not td
        assert new_td.shape == td.shape
        assert are_devices_equal(new_td.device, device)

        for leaf in pytree.tree_leaves(new_td):
            assert leaf.dtype == target_dtype
