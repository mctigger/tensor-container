import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import assert_tc_equal, run_and_compare_compiled
from tests.conftest import skipif_no_compile, skipif_no_cuda


@pytest.fixture
def base_td_data():
    """Provides raw data for a simple TensorDict."""
    return {
        "obs": torch.randn(4, 5, 3, 32, 32),
        "reward": torch.randn(4, 5, 1),
    }


@pytest.fixture
def simple_td(base_td_data):
    """A simple TensorDict instance."""
    return TensorDict(base_td_data, shape=(4, 5), device="cpu")


@pytest.fixture
def nested_td():
    """A TensorDict with nested structures."""
    B, T = 2, 3
    return TensorDict(
        {
            "obs": torch.randn(B, T, 10),
            "nested": TensorDict(
                {"state": torch.randn(B, T, 5)}, shape=(B, T), device="cpu"
            ),
        },
        shape=(B, T),
        device="cpu",
    )


@pytest.mark.skipif_no_compile
class TestTensorDictCompilation:
    """
    Tests for TensorDict operations under torch.compile.
    """

    @skipif_no_compile
    def test_creation_from_tensors_in_compiled_function(self, base_td_data):
        """
        Verifies that a TensorDict can be successfully created from raw tensors
        within a torch.compile'd function.
        """

        def create_td_from_tensors(obs_arg, reward_arg):
            td = TensorDict(
                {"obs": obs_arg, "reward": reward_arg},
                shape=(4, 5),
                device="cpu",
            )
            return td

        obs, reward = base_td_data["obs"], base_td_data["reward"]
        run_and_compare_compiled(create_td_from_tensors, obs, reward)

    @skipif_no_compile
    def test_creation_from_nested_dict_in_compiled_function(self, base_td_data):
        """
        Verifies that a TensorDict can be successfully created from a nested dictionary
        within a torch.compile'd function.
        """

        def create_td_from_nested_dict(obs_arg, reward_arg):
            td = TensorDict(
                {"my_dict": {"obs": obs_arg, "reward": reward_arg}},
                shape=(4, 5),
                device="cpu",
            )
            return td

        obs, reward = base_td_data["obs"], base_td_data["reward"]
        run_and_compare_compiled(create_td_from_nested_dict, obs, reward)

    @skipif_no_compile
    def test_setting_values_in_compiled_function(self, simple_td):
        """
        Tests that setting and modifying values within a TensorDict
        works correctly inside a compiled function.
        """

        def set_value_and_return_td(td_arg):
            # Perform the modification
            td_arg["reward"] = td_arg["reward"] * 2.0
            # Return the TensorDict that was (conceptually) modified
            return td_arg

        compiled_fn = torch.compile(set_value_and_return_td, fullgraph=True)

        # Clone for independent runs
        simple_td_eager = simple_td.clone()
        simple_td_for_compile_input = simple_td.clone()

        # Eager run: simple_td_eager is modified in-place, and also returned.
        eager_modified_td = set_value_and_return_td(simple_td_eager)

        # Compiled run: simple_td_for_compile_input is passed.
        # compiled_fn returns the new, modified TensorDict created internally.
        # The original simple_td_for_compile_input object in this scope is not mutated.
        compiled_modified_td = compiled_fn(simple_td_for_compile_input)

        # 1. Compare the TensorDicts that reflect the modification.
        #    eager_modified_td is the same object as simple_td_eager (which was modified).
        #    compiled_modified_td is the new TensorDict returned by the compiled function.
        assert_tc_equal(eager_modified_td, compiled_modified_td)

        # 2. Verify that the input TensorDict to the compiled function (`simple_td_for_compile_input`)
        #    is mutated to reflect the changes, consistent with the returned TensorDict.
        #    This is an observed behavior when a PyTree input is modified and returned by torch.compile.
        assert_tc_equal(compiled_modified_td, simple_td_for_compile_input)

    @skipif_no_compile
    def test_stacking_tensordicts_in_compiled_function(self, simple_td):
        """
        Tests torch.stack operation on TensorDicts within a compiled function.
        """
        td_a = simple_td.clone()
        td_b = simple_td.clone()
        td_b["reward"] += 1.0  # Make it different for stacking

        def stack_tensordicts(d1, d2):
            return torch.stack([d1, d2], dim=0)

        eager_result, compiled_result = run_and_compare_compiled(
            stack_tensordicts, td_a, td_b
        )
        assert eager_result.shape == (2, 4, 5)

    @skipif_no_compile
    def test_concatenating_tensordicts_in_compiled_function(self, simple_td):
        """
        Tests torch.cat operation on TensorDicts within a compiled function.
        """
        td_a = simple_td.clone()
        td_b = simple_td.clone()

        def concatenate_tensordicts(d1, d2):
            return torch.cat([d1, d2], dim=1)

        eager_result, compiled_result = run_and_compare_compiled(
            concatenate_tensordicts, td_a, td_b
        )
        assert eager_result.shape == (4, 10)

    @pytest.mark.parametrize(
        "index",
        [
            0,
            slice(1, 3),
            torch.tensor([0, 3]),  # advanced indexing
        ],
    )
    @skipif_no_compile
    def test_indexing_tensordict_in_compiled_function(self, simple_td, index):
        """
        Tests various forms of indexing a TensorDict within a compiled function.
        """

        def index_tensordict(td_arg):
            return td_arg[index]

        run_and_compare_compiled(index_tensordict, simple_td)

    @skipif_no_compile
    def test_operations_on_nested_tensordict_in_compiled_function(self, nested_td):
        """
        Tests slicing and modifying a nested TensorDict within a compiled function.
        """

        def operate_on_nested_td(td):
            sliced_td = td[0]
            sliced_td["nested"]["state"] = sliced_td["nested"]["state"] + 5.0
            return sliced_td

        compiled_fn = torch.compile(operate_on_nested_td, fullgraph=True)

        # Clone to ensure independent eager and compiled runs
        nested_td_eager = nested_td.clone()
        nested_td_compiled = nested_td.clone()

        eager_result = operate_on_nested_td(nested_td_eager)
        compiled_result = compiled_fn(nested_td_compiled)

        assert eager_result.shape == (3,)
        assert eager_result["nested"].shape == (3,)
        assert_tc_equal(eager_result, compiled_result)

    @skipif_no_compile
    @skipif_no_cuda
    def test_compile_to_device(self, simple_td):
        """
        Tests moving a TensorDict to a CUDA device within a compiled function.
        """

        def move_td_to_cuda(td_arg):
            return td_arg.to("cuda")

        eager_result, compiled_result = run_and_compare_compiled(
            move_td_to_cuda, simple_td
        )
        assert eager_result.device.type == "cuda"

    @skipif_no_compile
    def test_dtype_change_in_compiled_function(self, simple_td):
        """
        Tests changing the dtype of a TensorDict's tensors within a compiled function.
        """

        def change_td_dtype(td_arg):
            return td_arg.to(torch.float16)

        eager_result, compiled_result = run_and_compare_compiled(
            change_td_dtype, simple_td
        )
        assert eager_result["obs"].dtype == torch.float16
