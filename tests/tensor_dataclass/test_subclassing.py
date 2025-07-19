import inspect
from typing import Optional

import pytest
import torch

from tensorcontainer.tensor_dataclass import TensorDataClass


def _assert_init_signature(cls, expected_fields):
    actual_signature = inspect.signature(cls.__init__)

    parameters = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter(
            "shape",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=torch.Size,
        ),
        inspect.Parameter(
            "device",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Optional[torch.device],
        ),
    ]
    for field_name, field_type in expected_fields.items():
        parameters.append(
            inspect.Parameter(
                field_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=field_type,
            )
        )
    expected_signature = inspect.Signature(parameters, return_annotation=None)
    assert actual_signature == expected_signature


@pytest.fixture
def single_parent_class():
    class A(TensorDataClass):
        x: torch.Tensor

    class B(A):
        y: str

    return B


@pytest.fixture
def multiple_tensor_data_class_parents():
    class A(TensorDataClass):
        x: torch.Tensor

    class B(TensorDataClass):
        y: str

    class C(B, A):
        z: int

    return C


@pytest.fixture
def single_tensor_data_class_parent():
    class A(TensorDataClass):
        x: torch.Tensor

    class B:
        y: str

    class C(A, B):
        z: int

    return C


class TestInitSignature:
    def test_single_parent_class(self, single_parent_class):
        _assert_init_signature(single_parent_class, {"x": torch.Tensor, "y": str})

    def test_multiple_parent_classes_multiple_tensor_data_class(
        self, multiple_tensor_data_class_parents
    ):
        _assert_init_signature(
            multiple_tensor_data_class_parents,
            {"x": torch.Tensor, "y": str, "z": int},
        )

    def test_multiple_parent_classes_single_tensor_data_class(
        self, single_tensor_data_class_parent
    ):
        _assert_init_signature(
            single_tensor_data_class_parent, {"x": torch.Tensor, "z": int}
        )


class TestProperties:
    def test_property_evaluation(self):
        class A(TensorDataClass):
            x: torch.Tensor

            @property
            def y(self):
                return self.x + 1

        instance = A(
            x=torch.tensor(10), shape=torch.Size([]), device=torch.device("cpu")
        )
        assert instance.y == 11

    def test_property_not_in_init_signature(self):
        class A(TensorDataClass):
            x: torch.Tensor

            @property
            def y(self):
                return self.x + 1

        actual_signature = inspect.signature(A.__init__)
        assert "y" not in actual_signature.parameters

    def test_single_parent_class_instantiation(self, single_parent_class):
        instance = single_parent_class(
            x=torch.ones(1),
            y="hello",
            shape=torch.Size([1]),
            device=torch.device("cpu"),
        )
        assert torch.equal(instance.x, torch.ones(1))
        assert instance.y == "hello"

    def test_multiple_parent_classes_multiple_tensor_data_class_instantiation(
        self, multiple_tensor_data_class_parents
    ):
        x = torch.ones(1)
        y = "world"
        z = 123
        instance = multiple_tensor_data_class_parents(
            x=x,
            y=y,
            z=z,
            shape=torch.Size([1]),
            device=torch.device("cpu"),
        )
        assert instance.x is x
        assert instance.y == y
        assert instance.z == z

    def test_multiple_parent_classes_single_tensor_data_class_instantiation(
        self, single_tensor_data_class_parent
    ):
        x = torch.ones(1)
        z = 123
        instance = single_tensor_data_class_parent(
            x=x,
            z=z,
            shape=torch.Size([1]),
            device=torch.device("cpu"),
        )
        assert instance.x is x
        assert instance.z == z
