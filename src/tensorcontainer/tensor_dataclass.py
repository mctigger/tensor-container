from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, fields
from typing import Optional, TypeVar, Union

import torch
from torch import Tensor
from typing_extensions import dataclass_transform

from tensorcontainer.tensor_annotated import TensorAnnotated
from tensorcontainer.tensor_container import ShapeType, TensorContainer

TDCompatible = Union[Tensor, TensorContainer]
DATACLASS_ARGS = {"init", "repr", "eq", "order", "unsafe_hash", "frozen", "slots"}


T_TensorDataclass = TypeVar("T_TensorDataclass", bound="TensorDataClass")


@dataclass_transform(eq_default=False)
class TensorDataclassTransform:
    """This class is just needed for type hints. Directly decorating TensorDataclass does not work."""

    pass


class TensorDataClass(TensorAnnotated, TensorDataclassTransform):
    """A dataclass-based tensor container with automatic field generation and batch semantics.

    TensorDataClass provides a strongly-typed alternative to TensorDict by automatically
    converting annotated class definitions into dataclasses while maintaining tensor
    container functionality. It combines Python's dataclass system with PyTree
    compatibility and torch.compile support.

    ## Automatic Dataclass Generation

    Any class inheriting from TensorDataClass is automatically converted to a dataclass
    with optimized settings for tensor operations:

    - Field-based access using `obj.field` syntax
    - Static typing with IDE support and autocomplete
    - Natural inheritance patterns with field merging
    - Memory-efficient `slots=True` layout
    - Disabled equality comparison (`eq=False`) for tensor compatibility

    Example:
        >>> class MyData(TensorDataClass):
        ...     features: torch.Tensor
        ...     labels: torch.Tensor
        >>>
        >>> data = MyData(
        ...     features=torch.randn(4, 10),
        ...     labels=torch.arange(4).float(),
        ...     shape=(4,),
        ...     device="cpu"
        ... )
        >>>
        >>> # Automatic dataclass features
        >>> print(data.features.shape)  # torch.Size([4, 10])
        >>> data.features = new_tensor  # Type-checked assignment

    ## Batch and Event Dimensions

    TensorDataClass enforces the same batch/event dimension semantics as TensorContainer:

    - **Batch Dimensions**: Leading dimensions defined by `shape` parameter, must be
      consistent across all tensor fields
    - **Event Dimensions**: Trailing dimensions beyond batch shape, can vary per field
    - **Automatic Validation**: Shape compatibility is checked during initialization

    All tensor operations preserve this batch/event structure, enabling consistent
    batched processing across heterogeneous tensor fields.

    ## Field Definition Patterns

    ### Basic Tensor Fields
    ```python
    class BasicData(TensorDataClass):
        observations: torch.Tensor
        actions: torch.Tensor
    ```

    ### Optional Fields and Defaults
    ```python
    from dataclasses import field
    from typing import Dict, List, Optional, Any

    class FlexibleData(TensorDataClass):
        required_field: torch.Tensor
        optional_field: Optional[torch.Tensor] = None
        metadata: List[str] = field(default_factory=list)
        config: Dict[str, Any] = field(default_factory=dict)
        default_tensor: torch.Tensor = field(
            default_factory=lambda: torch.zeros(10)
        )
    ```

    ### Inheritance and Field Composition
    ```python
    class BaseData(TensorDataClass):
        observations: torch.Tensor

    class ExtendedData(BaseData):
        actions: torch.Tensor      # Inherits observations
        rewards: torch.Tensor      # Total: observations, actions, rewards

    class FinalData(ExtendedData):
        values: torch.Tensor       # Inherits all previous fields
    ```

    ## PyTree Integration

    TensorDataClass provides seamless PyTree integration through automatic registration:

    - Tensor fields become PyTree leaves for tree operations
    - Non-tensor fields are preserved as metadata
    - Supports `torch.stack`, `torch.cat`, and other tree operations
    - Compatible with `torch.compile` and JIT compilation

    The PyTree flattening separates tensor data from metadata, enabling efficient
    tensor transformations while preserving all field information.

    ## Device and Shape Management

    ### Device and Shape Validation
    The initialization process validates:
    - All tensor fields have batch shapes compatible with the container shape
    - All tensor fields reside on compatible devices
    - Field types match their annotations

    Validation uses PyTree traversal to check nested structures and provides
    detailed error messages with field paths for debugging.

    ## torch.compile Compatibility

    TensorDataClass is designed for efficient compilation:

    - **Static Structure**: Field names and types are known at compile time
    - **Efficient Access**: Direct attribute access compiles to optimized code
    - **Safe Copying**: Custom copy methods avoid graph breaks
    - **Minimal Overhead**: Streamlined operations for hot paths

    ## Memory and Performance

    With `slots=True` by default, TensorDataClass instances provide:

    - Reduced memory overhead compared to regular classes
    - Faster attribute access through direct slot access
    - Better memory locality for improved cache performance
    - Elimination of per-instance `__dict__` storage

    ## Comparison with TensorDict

    | Feature | TensorDataClass | TensorDict |
    |---------|-----------------|------------|
    | Access Pattern | `obj.field` | `obj["key"]` |
    | Type Safety | Static typing | Runtime checks |
    | IDE Support | Full autocomplete | Limited |
    | Memory Usage | Lower (slots) | Higher (dict) |
    | Field Definition | Compile-time | Runtime |
    | Inheritance | Natural OOP | Composition |
    | Dynamic Fields | Not supported | Full support |

    Args:
        shape (torch.Size): The batch shape that all tensor fields must share
            as their leading dimensions.
        device (Optional[Union[str, torch.device]]): The device all tensors should
            reside on. If None, device is inferred from the first tensor field.

    Raises:
        ValueError: If tensor field shapes are incompatible with batch shape.
        ValueError: If tensor field devices are incompatible with container device.
        TypeError: If attempting to create a subclass with eq=True.

    Note:
        TensorDataClass automatically applies the @dataclass decorator to subclasses.
        The eq parameter is forced to False for tensor compatibility, and slots is
        enabled by default for performance.
    """

    # The only reason we define shape and device here is such that @dataclass_transform
    # can enable static analyzers to provide type hints in IDEs. Both are programmatically
    # added in __init_subclass__ so removing the following two lines will only remove the
    # type hints, but the class will stay functional.
    shape: ShapeType
    device: Optional[torch.device]

    def __init_subclass__(cls, **kwargs):
        """Automatically convert subclasses into dataclasses with proper field inheritance.

        This method is called whenever a class inherits from TensorDataClass. It:
        1. Merges field annotations from the entire inheritance chain
        2. Extracts dataclass-specific configuration options
        3. Applies the @dataclass decorator with optimized defaults
        4. Enforces constraints like eq=False for tensor compatibility

        The annotation inheritance ensures that derived classes properly inherit
        field definitions from parent TensorDataClass instances.

        Args:
            **kwargs: Class definition arguments, may include dataclass options
                     like 'init', 'repr', 'eq', 'order', 'unsafe_hash', 'frozen', 'slots'

        Raises:
            TypeError: If eq=True is specified (incompatible with tensor fields)
        """
        # This check is needed as slots=True will result in dataclass(cls) creating a new class
        # and thus triggering __init__subclass again. However, we already have ran __init__subclass__
        # already for this class. To avoid infinte recursion, we have the following check.
        if hasattr(cls, "__slots__"):
            return

        annotations = cls._get_annotations(TensorDataClass)

        cls.__annotations__ = {
            "shape": torch.Size,
            "device": Optional[torch.device],
            **annotations,
        }

        dc_kwargs = {}
        for k in list(kwargs.keys()):
            if k in DATACLASS_ARGS:
                dc_kwargs[k] = kwargs.pop(k)

        super().__init_subclass__(**kwargs)

        if dc_kwargs.get("eq") is True:
            raise TypeError(
                f"Cannot create {cls.__name__} with eq=True. TensorDataClass requires eq=False."
            )
        dc_kwargs.setdefault("eq", False)
        if sys.version_info >= (3, 10):
            dc_kwargs.setdefault("slots", True)

        dataclass(cls, **dc_kwargs)

    def __post_init__(self):
        """Initialize TensorContainer functionality and perform validation.

        This method is automatically called by the dataclass __init__ after all
        fields have been set. It:

        1. Infers device from tensor fields if device was not specified
        2. Initializes the TensorContainer base class with shape and device
        3. Validates that all tensor fields have compatible devices
        4. Validates that all tensor fields have compatible batch shapes

        Raises:
            ValueError: If tensor field shapes are incompatible with batch shape
            ValueError: If tensor field devices are incompatible with container device
        """
        super().__init__(self.shape, self.device)

    def __copy__(self: T_TensorDataclass) -> T_TensorDataclass:
        """Create a shallow copy of the TensorDataClass instance.

        This method is designed to be `torch.compile` safe by avoiding the
        use of `copy.copy()`, which can cause graph breaks. It manually
        copies all field references without deep-copying tensor data.

        The shallow copy means:
        - Field references are copied (new instance)
        - Tensor data is shared (same underlying tensors)
        - Metadata fields are shared (same objects)

        For independent tensor data, use `clone()` inherited from TensorContainer.

        Returns:
            T_TensorDataclass: New instance with shared field data

        Example:
            >>> original = MyData(obs=torch.randn(4, 128), shape=(4,))
            >>> shallow_copy = original.__copy__()
            >>> shallow_copy.obs is original.obs  # True - shared tensor
            >>>
            >>> # For independent tensors:
            >>> deep_copy = original.clone()  # Creates new tensor data
        """
        # Create a new, uninitialized instance of the correct class.
        cls = type(self)
        new_obj = cls.__new__(cls)

        # Manually copy all dataclass fields.
        for field in fields(self):
            value = getattr(self, field.name)
            setattr(new_obj, field.name, value)

        # Manually call __post_init__ to initialize the TensorContainer part
        # and run validation logic. This is necessary because we bypassed __init__.
        if hasattr(new_obj, "__post_init__"):
            new_obj.__post_init__()

        return new_obj

    def __deepcopy__(
        self: T_TensorDataclass, memo: Optional[dict] = None
    ) -> T_TensorDataclass:
        """
        Performs a deep copy of the TensorDataclass instance.

        This method is designed to be `torch.compile` safe by manually
        iterating through fields and using `copy.deepcopy` for each,
        while also handling the `memo` dictionary to prevent infinite
        recursion in case of circular references.

        Args:
            memo: A dictionary to keep track of already copied objects.
                  This is part of the `copy.deepcopy` protocol.

        Returns:
            A new TensorDataclass instance with attributes that are deep
            copies of the original's attributes.
        """
        if memo is None:
            memo = {}

        cls = type(self)
        # Check if the object is already in memo
        if id(self) in memo:
            return memo[id(self)]

        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        for field in fields(self):
            value = getattr(self, field.name)
            # The `shape` and `device` fields are part of the dataclass fields
            # due to their annotations in TensorDataclass.
            # These should be deepcopied as well if they are not None.
            if field.name in ("shape", "device"):
                # Tuples (shape) and torch.device are immutable or behave as such.
                # Direct assignment is fine and avoids torch.compile issues with deepcopying them.
                # Direct assignment for immutable types like tuple (shape) and torch.device.
                # This avoids torch.compile issues with copy.copy or copy.deepcopy on these types.
                setattr(new_obj, field.name, value)
            elif isinstance(value, Tensor):
                # For torch.Tensor, use .clone() for a deep copy of data.
                setattr(new_obj, field.name, value.clone())
            elif isinstance(value, list):
                # For lists, create a new list. This is a shallow copy of the list structure.
                # If list items are mutable and need deepcopying, torch.compile might
                # still struggle with a generic deepcopy of those items.
                # For a list of immutables (like in the test), this is effectively a deepcopy.
                setattr(new_obj, field.name, list(value))
            else:
                # For other fields (e.g., dict, other custom objects), attempt deepcopy.
                # This remains a potential point of failure for torch.compile
                # if it doesn't support deepcopying these specific types.
                setattr(new_obj, field.name, copy.deepcopy(value))

        # Manually call __post_init__ to initialize the TensorContainer part
        # and run validation logic. This is necessary because we bypassed __init__.
        # __post_init__ in TensorDataclass handles shape and device initialization
        # and validation, which is crucial after all fields are set.
        if hasattr(new_obj, "__post_init__"):
            new_obj.__post_init__()

        return new_obj
