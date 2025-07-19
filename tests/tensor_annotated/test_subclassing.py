from typing import Optional

import pytest
import torch
from torch import Tensor

from tensorcontainer.tensor_annotated import TensorAnnotated


class TestSubclassingPositive:
    """Tests for positive scenarios in TensorAnnotated subclassing.

    These tests ensure that subclassing TensorAnnotated works as expected
    when attributes are correctly annotated and inherited.
    """

    def test_single_parent_class_instantiation(self):
        """Tests subclassing with a single TensorAnnotated parent."""

        class A(TensorAnnotated):
            x: Tensor

            def __init__(self, x: Tensor):
                self.x = x
                super().__init__(shape=x.shape, device=x.device)

        class B(A):
            y: Tensor

            def __init__(
                self, x: Tensor, y: Tensor, **kwargs
            ):  # **kwargs captures device and shape in __init_from_reconstructed
                self.y = y
                super().__init__(x=x)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        instance = B(x=x, y=y)
        instance = instance.copy()  # To trigger pytree operations
        assert instance.x is x
        assert instance.y is y

    def test_multiple_parent_classes_multiple_tensor_annotated_instantiation(self):
        """Tests subclassing with multiple TensorAnnotated parents."""

        class A(TensorAnnotated):
            x: Tensor

            def __init__(self, *, x: torch.Tensor, **kwargs):
                self.x = x
                super().__init__(**kwargs)

        class B(TensorAnnotated):
            y: Tensor

            def __init__(self, *, y: torch.Tensor, **kwargs):
                self.y = y
                super().__init__(**kwargs)

        class C(B, A):
            z: Tensor

            def __init__(
                self, x, y, z, **kwargs
            ):  # **kwargs captures device and shape in __init_from_reconstructed
                self.z = z
                super().__init__(shape=x.shape, device=x.device, x=x, y=y)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        z = torch.randn(2, 3, 4)
        instance = C(x=x, y=y, z=z)
        instance = instance.copy()  # To trigger pytree operations
        assert instance.x is x
        assert instance.y is y
        assert instance.z is z

    def test_multiple_parent_classes_single_tensor_annotated_instantiation(self):
        """Tests subclassing with one TensorAnnotated parent and other non-TensorAnnotated parents."""

        class A(TensorAnnotated):
            x: Tensor

            def __init__(self, x: Tensor):
                self.x = x
                super().__init__(shape=x.shape, device=x.device)

        class B:
            pass

        class C(A, B):
            z: Tensor

            def __init__(
                self, x: torch.Tensor, z: Tensor, **kwargs
            ):  # **kwargs captures device and shape in __init_from_reconstructed
                self.z = z
                super().__init__(x=x)

        x = torch.randn(2, 3, 4)
        z = torch.randn(2, 3, 4)
        instance = C(
            x=x,
            z=z,
        )
        instance = instance.copy()  # To trigger pytree operations
        assert instance.x is x
        assert instance.z is z


class TestSubclassingNegative:
    """Tests for negative scenarios in TensorAnnotated subclassing.

    These tests ensure that attributes are correctly ignored or handled
    when they do not conform to the TensorAnnotated structure, such as
    attributes from non-TensorAnnotated parent classes or non-annotated attributes.
    """

    def test_attribute_not_part_of_flatten(self):
        """Tests that attributes from non-TensorAnnotated parent classes are ignored."""

        class A(TensorAnnotated):
            x: Tensor

            def __init__(self, x: Tensor, **kwargs):
                self.x = x
                init_shape = kwargs.pop("shape", x.shape)
                init_device = kwargs.pop("device", x.device)
                super().__init__(shape=init_shape, device=init_device, **kwargs)

        class B:
            def __init__(self, y: Optional[Tensor] = None):
                if y is not None:
                    self.y = y

        class C(A, B):
            z: Tensor

            def __init__(
                self, x: Tensor, z: Tensor, y: Optional[Tensor] = None, **kwargs
            ):
                self.z = z
                A.__init__(self, x=x, **kwargs)
                B.__init__(self, y=y)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        z = torch.randn(2, 3, 4)
        instance = C(x=x, y=y, z=z)
        instance_copy = instance.copy()

        # x and z should be present, y should not
        assert instance_copy.x is instance.x
        assert instance_copy.z is instance.z
        with pytest.raises(AttributeError, match="object has no attribute 'y'"):
            instance_copy.y

    def test_no_annotation(self):
        """Tests that non-annotated attributes are ignored."""

        class A(TensorAnnotated):
            x: Tensor

            def __init__(self, x: Tensor, **kwargs):
                self.x = x
                init_shape = kwargs.pop("shape", x.shape)
                init_device = kwargs.pop("device", x.device)
                super().__init__(shape=init_shape, device=init_device, **kwargs)

        class B(A):
            def __init__(self, x: Tensor, y: Optional[Tensor] = None, **kwargs):
                if y is not None:
                    self.y = y  # y is not annotated
                super().__init__(x=x, **kwargs)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        instance = B(x=x, y=y)
        instance_copy = instance.copy()

        assert instance_copy.x is instance.x
        with pytest.raises(AttributeError, match="object has no attribute 'y'"):
            instance_copy.y

    def test_only_child_tensor_annotated(self):
        """Tests that only attributes from the direct TensorAnnotated subclass are used."""

        class A:
            def __init__(self, x: Tensor):
                self.x = x

        class B:
            def __init__(self, y: Tensor):
                self.y = y

        class C(A, B, TensorAnnotated):
            z: Tensor

            def __init__(
                self,
                z: Tensor,
                x: Optional[Tensor] = None,
                y: Optional[Tensor] = None,
                **kwargs,
            ):
                if x is not None:
                    A.__init__(self, x=x)
                if y is not None:
                    B.__init__(self, y=y)
                self.z = z
                init_shape = kwargs.pop("shape", z.shape)
                init_device = kwargs.pop("device", z.device)
                super(B, self).__init__(shape=init_shape, device=init_device, **kwargs)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        z = torch.randn(2, 3, 4)
        instance = C(x=x, y=y, z=z)
        instance_copy = instance.copy()

        assert instance_copy.z is instance.z
        with pytest.raises(AttributeError, match="object has no attribute 'x'"):
            instance_copy.x
        with pytest.raises(AttributeError, match="object has no attribute 'y'"):
            instance_copy.y
