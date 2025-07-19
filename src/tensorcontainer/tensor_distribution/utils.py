from typing import Tuple

import torch
from torch import Tensor


def _process_args(
    *args: Tensor,
) -> Tuple[torch.Size, torch.Size, torch.dtype, torch.device]:
    """
    Processes a sequence of tensors to determine their common batch shape, event shape, dtype, and device.

    Args:
        *args: A sequence of tensors.

    Returns:
        A tuple containing:
            - batch_shape (torch.Size): The common batch shape.
            - event_shape (torch.Size): The common event shape (derived from the last two dimensions for matrices).
            - dtype (torch.dtype): The common data type.
            - device (torch.device): The common device.
    """
    if not args:
        return (
            torch.Size(),
            torch.Size(),
            torch.get_default_dtype(),
            torch.device("cpu"),
        )

    non_none_args = [arg for arg in args if arg is not None]  # type: ignore
    if not non_none_args:
        return (
            torch.Size(),
            torch.Size(),
            torch.get_default_dtype(),
            torch.device("cpu"),
        )

    # Determine common dtype and device
    dtype = non_none_args[0].dtype
    device = non_none_args[0].device
    for arg in non_none_args:
        if arg.dtype != dtype:
            # Promote dtype if necessary (simple promotion for now, can be more sophisticated)
            if arg.dtype == torch.float64 or dtype == torch.float64:
                dtype = torch.float64
            elif arg.dtype == torch.float32 or dtype == torch.float32:
                dtype = torch.float32
        if arg.device != device:
            # If devices differ, it's an error or requires explicit handling
            # For now, assume all tensors are on the same device or will be moved
            pass

    # Determine common batch shape from all tensors
    batch_shapes = [
        arg.shape[:-2] if arg.ndim >= 2 else arg.shape for arg in non_none_args
    ]
    batch_shape = torch.broadcast_shapes(*batch_shapes)

    # Determine common event shape only from tensors that are matrices (ndim >= 2)
    matrix_args = [arg for arg in non_none_args if arg.ndim >= 2]
    if len(matrix_args) > 0:
        event_shapes = [arg.shape[-2:] for arg in matrix_args]
        first_event_shape = event_shapes[0]
        for es in event_shapes:
            if es != first_event_shape:
                raise ValueError(
                    "Inconsistent event shapes among matrix input tensors."
                )
        event_shape = first_event_shape
    else:
        event_shape = torch.Size()  # No matrix inputs, so no event shape

    return batch_shape, event_shape, dtype, device
