import torch

from tensorcontainer.tensor_dict import TensorDict


def test_items():
    data = {
        "a": torch.tensor([1, 2, 3]),
        "b": torch.tensor([4, 5, 6]),
    }
    td = TensorDict(data, shape=torch.Size([3]))
    items = td.items()
    expected_items = data.items()
    assert set(items) == set(expected_items)


def test_items_empty():
    td = TensorDict({}, shape=torch.Size([0]))
    items = td.items()
    assert set(items) == set()


def test_items_nested():
    data = {
        "a": torch.tensor([1, 2, 3]),
        "b": {"c": torch.tensor([4, 5, 6])},
    }
    td = TensorDict(data, shape=torch.Size([3]))
    items = list(td.items())
    expected_items = list(data.items())
    # The nested dict is converted to TensorDict, so we need to compare the keys and values
    assert set(k for k, v in items) == set(data.keys())
    assert items[1][0] == expected_items[1][0]
    items_b_values = list(
        TensorDict(expected_items[1][1], shape=torch.Size([3])).values()
    )
    assert torch.equal(list(items[1][1].values())[0], items_b_values[0])
