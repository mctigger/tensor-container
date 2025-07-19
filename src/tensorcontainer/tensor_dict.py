from __future__ import annotations

import functools
from typing import (
    Any,
    Dict,
    Iterable,  # Added for _pytree_unflatten signature
    List,
    Mapping,
    NamedTuple,  # Added for specialized context type
    Optional,
    Tuple,
    Union,
    cast,  # Added cast for explicit type hinting
    overload,  # Added cast for explicit type hinting
)

import torch

# Use the official PyTree utility from torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch.utils._pytree import (
    KeyEntry,
    MappingKey,
    PyTree,
)  # Explicitly imported for clarity and Pylance
from typing_extensions import TypeAlias

from tensorcontainer.tensor_container import TensorContainer
from tensorcontainer.utils import PytreeRegistered

TDCompatible: TypeAlias = Union[Tensor, TensorContainer]
NestedTDCompatible: TypeAlias = Union[TDCompatible, Dict[str, TDCompatible]]


# Define a NamedTuple for the pytree context to provide explicit typing
class TensorDictPytreeContext(NamedTuple):
    keys: Tuple[str, ...]
    event_ndims: Tuple[int, ...]
    shape_context: Tuple[int, ...]
    device_context: Optional[Union[str, torch.device]]


HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for TensorDict."""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class TensorDict(TensorContainer, PytreeRegistered):
    """
    Dictionary-like container for batched Tensors sharing a common batch shape.

    - PyTree & torch.compile compatible
    - Standard mapping ops: getitem, setitem, update, etc.
    - Utilities: flatten_keys, copy, and more

    Example:
        >>> td = TensorDict({'x': torch.zeros(4, 3)}, shape=(4,))
        >>> td['x'].shape
        torch.Size([4, 3])
        >>> td.flatten_keys()
        TensorDict(shape=(4,), x: Tensor(shape=(4,3)))
    """

    def __init__(
        self,
        data: Mapping[str, NestedTDCompatible],
        shape: Tuple[int, ...],
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initializes the TensorDict. This constructor is kept simple for
        `torch.compile` compatibility, performing direct attribute assignment.
        """
        self.data = TensorDict.data_from_dict(data, shape, device)

        super().__init__(shape, device)

    @classmethod
    def data_from_dict(cls, data, shape, device=None) -> Dict[str, TDCompatible]:
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                result[k] = TensorDict(
                    TensorDict.data_from_dict(v, shape, device), shape, device
                )
            else:
                result[k] = v

        return result

    def _get_path_str(self, key_path):
        """Helper to construct path string from key_path, robust to torch.compile."""
        path_parts = []
        for k in key_path:
            if isinstance(k, tuple):  # Handle nested KeyPath tuples
                path_parts.append(self._get_path_str(k))
            elif hasattr(k, "key"):  # Access the 'key' attribute of the Key object
                path_parts.append(str(k.key))
            else:  # Fallback for unexpected elements
                path_parts.append(str(k))
        return ".".join(path_parts)

    def _tree_validate_shape(self, data):
        """
        Validates that the shapes of all nested tensors in the TensorDict start
        with the expected batch shape.

        This method recursively traverses the entire data structure.
        """
        keypath_leaf_pairs = pytree.tree_leaves_with_path(data)
        batch_shape = self.shape

        for key_path, leaf in keypath_leaf_pairs:
            path_str = self._get_path_str(key_path)

            if leaf.ndim > 0 and not self._is_shape_compatible(leaf.shape):
                raise ValueError(
                    f"Shape mismatch at '{path_str}': The tensor shape {leaf.shape} "
                    f"is not compatible with the TensorDict's batch shape {batch_shape}."
                )

    def _tree_validate_device(self, data):
        """
        Validates that the devices of all nested tensors in the TensorDict match
        the TensorDict's device if specified.
        """
        keypath_leaf_pairs = pytree.tree_leaves_with_path(data)

        for key_path, leaf in keypath_leaf_pairs:
            path_str = self._get_path_str(key_path)

            if not self._is_device_compatible(leaf.device):
                raise ValueError(
                    f"Device mismatch at '{path_str}': The tensor device {leaf.device} "
                    f"is not compatible with the TensorDict's device {self.device}."
                )

    def _get_pytree_context(
        self,
        keys: List[str],
        flat_leaves: List[TDCompatible],
    ) -> TensorDictPytreeContext:
        """
        Private helper to compute the pytree context for this TensorDict.
        The context captures metadata to reconstruct the TensorDict:
        keys, event_ndims, original shape, device, and non-tensor metadata.
        """
        batch_ndim = len(self.shape)
        event_ndims = tuple(leaf.ndim - batch_ndim for leaf in flat_leaves)
        return TensorDictPytreeContext(
            tuple(keys), event_ndims, self.shape, self.device
        )

    def _pytree_flatten(
        self,
    ) -> Tuple[List[TDCompatible], TensorDictPytreeContext]:
        """
        Flattens the TensorDict into its tensor leaves and static metadata
        by performing a shallow, one-level separation of tensor-like values
        from other metadata.
        """
        leaves: List[TDCompatible] = []
        keys: List[str] = []
        for key, value in self.data.items():
            leaves.append(value)
            keys.append(key)

        context = self._get_pytree_context(keys, leaves)
        return leaves, context

    def _pytree_flatten_with_keys_fn(
        self,
    ) -> tuple[list[tuple[KeyEntry, Any]], Any]:
        """
        Flattens the TensorDict into key-path/leaf pairs and a context, aligning
        with the shallow-flattening logic.
        """
        leaves, context = self._pytree_flatten()
        # Create (key, value) pairs for pytree compatibility.
        key_value_pairs = [
            (cast(KeyEntry, MappingKey(k)), cast(Any, v))
            for k, v in zip(context.keys, leaves)
        ]
        return key_value_pairs, context

    @classmethod
    def _pytree_unflatten(
        cls, leaves: Iterable[TDCompatible], context: TensorDictPytreeContext
    ) -> PyTree:
        """
        Reconstructs a TensorDict from leaves and a context, using a shallow
        unflattening approach that leverages keys and metadata for reconstruction.
        """
        # Unpack context using positional correspondence for clarity
        keys, event_ndims, shape_context, device_context = context

        obj = cls.__new__(cls)
        obj.device = device_context
        leaves_list = list(leaves)
        if not leaves_list:
            # Handle the empty case
            obj.data = {}
            obj.shape = shape_context
            return obj

        first_leaf = leaves_list[0]

        # Reconstruct the data dictionary from leaves and metadata
        data = dict(zip(keys, leaves_list))
        obj.data = data

        # Calculate new_shape based on transformed leaves and event_ndims
        if (
            event_ndims and event_ndims[0] == 0
        ):  # Leaf was a scalar or had only batch dimensions originally
            reconstructed_shape = first_leaf.shape
        elif event_ndims:  # Leaf had event dimensions originally
            reconstructed_shape = first_leaf.shape[: -event_ndims[0]]
        else:  # No leaves with event_ndims, use context
            reconstructed_shape = shape_context

        obj.shape = reconstructed_shape

        return obj

    # --- Standard MutableMapping methods ---
    @overload
    def __getitem__(self, key: str) -> TDCompatible: ...

    @overload
    def __getitem__(self, key: slice) -> TensorDict: ...

    @overload
    def __getitem__(self, key: Tensor) -> TensorDict: ...

    def __getitem__(self, key: Any) -> TDCompatible:
        if isinstance(key, str):
            return self.data[key]

        return super().__getitem__(key)

    @overload
    def __setitem__(self, key: str, value: TDCompatible): ...

    @overload
    def __setitem__(
        self,
        key: Union[slice, Tensor, int, Tuple[Union[slice, Tensor, int], ...]],
        value: Union[float, int, Tensor, TensorDict],
    ): ...

    def __setitem__(self, key: Any, value: Any):
        if isinstance(key, str):
            if isinstance(value, dict):
                value = TensorDict(value, self.shape, self.device)
            else:
                self._validate_device(value)
                self._validate_shape(value)

            self.data[key] = value
        else:
            # Handle slicing operations
            if isinstance(value, (float, int)):
                # Convert scalar to a tensor for assignment via TensorContainer's __setitem__
                value = torch.tensor(
                    value, device=self.device, dtype=torch.float32
                )  # Assuming float for scalar assignment

            super().__setitem__(key, value)

    def __delitem__(self, key: str):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def update(self, other: Union[Dict[str, TDCompatible], TensorDict]):
        """
        Updates the TensorDict with values from another dictionary or TensorDict.
        """
        if isinstance(other, TensorDict):
            other = other.data
        for key, value in other.items():
            self[key] = value

    def flatten_keys(self, separator: str = ".") -> TensorDict:
        """
        Returns a TensorDict with flattened keys using an iterative approach
        to avoid recursion and temporary reference cycles.
        """
        out = {}
        # Stack for iterative traversal: (data, prefix)
        stack: List[Tuple[TDCompatible, str]] = [(self, "")]

        while stack:
            data, prefix = stack.pop()

            if isinstance(data, TensorDict):
                for key, value in data.items():
                    new_prefix = prefix + key + separator
                    stack.append((value, new_prefix))
            else:
                # Store the flattened value with its full key
                out[prefix[:-1]] = data

        return TensorDict(out, self.shape, self.device)
