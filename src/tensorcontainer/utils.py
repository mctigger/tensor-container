from abc import abstractmethod
from typing import Any, Iterable, List, Tuple, Type, TypeVar, Union

import torch
import torch.utils._pytree as pytree
from torch._prims_common import DeviceLikeType
from torch.utils._pytree import Context, KeyEntry, PyTree

_PytreeRegistered = TypeVar("_PytreeRegistered", bound="PytreeRegistered")


class PytreeRegistered:
    """
    A mixin class that automatically registers any of its subclasses
    with the PyTorch PyTree system upon definition.
    """

    def __init_subclass__(cls, **kwargs):
        # This method is called by Python when a class that inherits
        # from PytreeRegistered is defined. `cls` is the new subclass.
        super().__init_subclass__(**kwargs)

        pytree.register_pytree_node(
            cls,
            cls._pytree_flatten,
            cls._pytree_unflatten,
            flatten_with_keys_fn=cls._pytree_flatten_with_keys_fn,
        )

    @abstractmethod
    def _pytree_flatten(self) -> Tuple[List[Any], Context]:
        pass

    @abstractmethod
    def _pytree_flatten_with_keys_fn(
        self,
    ) -> Tuple[List[Tuple[KeyEntry, Any]], Any]:
        pass

    @classmethod
    @abstractmethod
    def _pytree_unflatten(
        cls: Type[_PytreeRegistered], leaves: Iterable[Any], context: Context
    ) -> PyTree:
        pass


def resolve_device(device_str: Union[str, DeviceLikeType]) -> torch.device:
    """
    Dynamically resolves a device string to a torch.device object with a
    specific device index, if the backend is available and supports it.

    This works for any backend that follows the torch.cuda pattern,
    such as 'cuda', 'xpu', etc.
    """
    device = torch.device(device_str)

    # 1. If an index is already specified or it's a non-indexable device, return it.
    if device.index is not None or device.type in ["meta", "cpu"]:
        return device

    # 2. Dynamically check for the backend module (e.g., torch.cuda, torch.xpu).
    if hasattr(torch, device.type):
        backend = getattr(torch, device.type)

        # 3. Check if the backend is available and can report the current device.
        if (
            hasattr(backend, "is_available")
            and backend.is_available()
            and hasattr(backend, "current_device")
        ):
            current_index = backend.current_device()
            return torch.device(f"{device.type}:{current_index}")

    # 4. If the backend doesn't exist or isn't set up, return the original device.
    return device
