import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict


def test_reshape_basic():
    td = TensorDict(
        {"a": torch.randn(4, 4)},
        shape=(4, 4),
        device=torch.device("cpu"),
    )
    td_reshape = td.reshape(16)
    assert td_reshape.shape == (16,)
    assert torch.equal(td_reshape["a"], td["a"].reshape(16))
    assert td_reshape["a"].is_contiguous()


def test_reshape_reshape():
    td = TensorDict(
        {"a": torch.randn(4, 4)},
        shape=(4, 4),
        device=torch.device("cpu"),
    )
    td_reshape = td.reshape(2, 8)
    assert td_reshape.shape == (2, 8)
    assert torch.equal(td_reshape["a"], td["a"].reshape(2, 8))
    assert td_reshape["a"].is_contiguous()


def test_reshape_nested():
    td = TensorDict(
        {
            "a": TensorDict(
                {"b": torch.randn(2, 2)}, shape=(2, 2), device=torch.device("cpu")
            )
        },
        shape=(2, 2),
        device=torch.device("cpu"),
    )
    td_reshape = td.reshape(4)
    assert td_reshape.shape == (4,)
    assert td_reshape["a"]["b"].is_contiguous()
    assert td_reshape["a"]["b"].shape == (4,)


def test_reshape_multiple_keys():
    td = TensorDict(
        {"a": torch.randn(2, 2), "b": torch.ones(2, 2)},
        shape=(2, 2),
        device=torch.device("cpu"),
    )
    td_reshape = td.reshape(4)
    assert td_reshape.shape == (4,)
    assert td_reshape["a"].is_contiguous()
    assert td_reshape["b"].is_contiguous()
    assert td_reshape["a"].shape == (4,)
    assert td_reshape["b"].shape == (4,)


def test_reshape_nested_shape_validation():
    # Valid case: child shape is the same as the parent shape after reshape
    td1 = TensorDict(
        {
            "a": TensorDict(
                {"b": torch.randn(4)}, shape=(4,), device=torch.device("cpu")
            )
        },
        shape=(4,),
        device=torch.device("cpu"),
    )
    td1.reshape(4)

    # Invalid case: child shape is not compatible with the parent shape after reshape
    td2 = TensorDict(
        {
            "a": TensorDict(
                {"b": torch.randn(2, 2)}, shape=(2, 2), device=torch.device("cpu")
            )
        },
        shape=(2, 2),
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError):
        td2.reshape(5)


def test_reshape_invalid_shape():
    td = TensorDict({"a": torch.randn(2, 2)}, shape=(2, 2), device=torch.device("cpu"))
    with pytest.raises(RuntimeError):
        td.reshape(3)


def test_reshape_single_element_tensordict():
    td = TensorDict({"a": torch.tensor([1.0])}, shape=(1,), device=torch.device("cpu"))
    td_reshape = td.reshape(1)
    assert td_reshape.shape == (1,)
    assert torch.equal(td_reshape["a"], torch.tensor([1.0]))
    assert td_reshape["a"].is_contiguous()
    assert td_reshape["a"].shape == (1,)
