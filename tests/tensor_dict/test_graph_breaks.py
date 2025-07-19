import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.compile_utils import run_and_compare_compiled


@pytest.fixture
def sample_tensordict():
    return TensorDict({"a": torch.randn(3, 4), "b": torch.randn(3, 4, 5)}, shape=(3, 4))


@pytest.fixture
def sample_nested_tensordict():
    return TensorDict(
        {
            "a": torch.randn(3, 4),
            "b": TensorDict({"c": torch.randn(3, 4, 5)}, shape=(3, 4)),
        },
        shape=(3, 4),
    )


@pytest.mark.skipif_no_compile
class TestGraphBreaks:
    def test_init(self):
        def f(args, *kwargs):
            data, shape = args
            return TensorDict(data, shape)

        run_and_compare_compiled(
            f,
            (
                {"a": torch.randn(3, 4), "b": torch.randn(3, 4, 5)},
                (3, 4),
            ),
        )

    def test_getitem_str(self, sample_tensordict):
        def f(args, *kwargs):
            td, key = args
            return td[key]

        run_and_compare_compiled(f, (sample_tensordict, "a"))

    def test_getitem_slice(self, sample_tensordict):
        def f(args, *kwargs):
            td, idx = args
            return td[idx]

        run_and_compare_compiled(f, (sample_tensordict, 1))
        run_and_compare_compiled(f, (sample_tensordict, slice(0, 2)))

    def test_setitem(self, sample_tensordict):
        def f(args, *kwargs):
            td, key, value = args
            td_copy = td.clone()
            td_copy[key] = value
            return td_copy

        run_and_compare_compiled(f, (sample_tensordict, "c", torch.randn(3, 4)))

    def test_delitem(self, sample_tensordict):
        def f(args, *kwargs):
            td, key = args
            td_copy = td.clone()
            del td_copy[key]
            return td_copy

        td_copy = sample_tensordict.clone()
        run_and_compare_compiled(f, (td_copy, "a"))

    def test_iter(self, sample_tensordict):
        def f(args, *kwargs):
            td = args[0]
            return list(td.keys())

        run_and_compare_compiled(f, (sample_tensordict,))

    def test_len(self, sample_tensordict):
        def f(args, *kwargs):
            td = args[0]
            return len(td)

        run_and_compare_compiled(f, (sample_tensordict,))

    def test_contains(self, sample_tensordict):
        def f(args, *kwargs):
            td, key = args
            return key in td

        run_and_compare_compiled(f, (sample_tensordict, "a"))

    def test_keys(self, sample_tensordict):
        def f(args, *kwargs):
            td = args[0]
            return list(td.keys())

        run_and_compare_compiled(f, (sample_tensordict,))

    def test_values(self, sample_tensordict):
        def f(args, *kwargs):
            td = args[0]
            return list(td.values())

        run_and_compare_compiled(f, (sample_tensordict,))

    def test_items(self, sample_tensordict):
        def f(args, *kwargs):
            td = args[0]
            return list(td.items())

        run_and_compare_compiled(f, (sample_tensordict,))

    def test_view(self, sample_tensordict):
        def f(args, *kwargs):
            td, shape = args
            return td["a"].view(*shape)

        run_and_compare_compiled(f, (sample_tensordict, (3, 4)))

    def test_reshape(self, sample_tensordict):
        def f(args, *kwargs):
            td, shape = args
            return td["a"].reshape(*shape)

        run_and_compare_compiled(f, (sample_tensordict, (3, 4)))

    def test_to(self, sample_tensordict):
        def f(args, *kwargs):
            td, device = args
            return td.to(device)

        run_and_compare_compiled(f, (sample_tensordict, "cpu"))

    def test_detach(self, sample_tensordict):
        def f(args, *kwargs):
            td = args[0]
            return td.detach()

        run_and_compare_compiled(f, (sample_tensordict,))

    def test_clone(self, sample_tensordict):
        def f(args, *kwargs):
            td = args[0]
            return td.clone()

        run_and_compare_compiled(f, (sample_tensordict,))

    def test_expand(self, sample_tensordict):
        def f(args, *kwargs):
            td, shape = args
            return td["a"].expand(*shape)

        run_and_compare_compiled(f, (sample_tensordict, (3, 4)))

    def test_update(self, sample_tensordict):
        def f(args, *kwargs):
            td, other = args
            td_copy = td.clone()
            td_copy.update(other)
            return td_copy

        other_td = TensorDict({"c": torch.randn(3, 4)}, shape=(3, 4))
        run_and_compare_compiled(f, (sample_tensordict.clone(), other_td))
        run_and_compare_compiled(
            f, (sample_tensordict.clone(), {"d": torch.randn(3, 4)})
        )

    def test_copy(self, sample_nested_tensordict):
        def f(args, *kwargs):
            td = args[0]
            return td.copy()

        run_and_compare_compiled(f, (sample_nested_tensordict,))

    def test_flatten_keys(self, sample_nested_tensordict):
        def f(args, *kwargs):
            td = args[0]
            return td.flatten_keys()

        run_and_compare_compiled(f, (sample_nested_tensordict,))

    def test_setitem_tensor(self, sample_tensordict):
        def f(args, *kwargs):
            td, key, value = args
            td[key] = value
            return td

        td = sample_tensordict.clone()
        value = torch.randn(3, 4)
        run_and_compare_compiled(f, (td, "a", value))
