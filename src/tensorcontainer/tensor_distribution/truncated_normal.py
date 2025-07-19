from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Size, Tensor
from torch.distributions import Distribution, Independent

from tensorcontainer.distributions.truncated_normal import TruncatedNormal

from .base import TensorDistribution


class TensorTruncatedNormal(TensorDistribution):
    _loc: Tensor
    _scale: Tensor
    _low: Tensor
    _high: Tensor
    _validate_args: Optional[bool] = None
    _event_shape: Size

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        low: Tensor,
        high: Tensor,
        validate_args: bool | None = None,
    ):
        loc, scale, low, high = torch.broadcast_tensors(loc, scale, low, high)
        self._loc = loc
        self._scale = scale
        self._low = low
        self._high = high
        self._validate_args = validate_args

        # Determine batch_shape and event_shape
        # Assuming the last dimension of loc, scale, low, high is the event dimension
        # and the rest are batch dimensions.
        # If loc is a scalar, batch_shape is () and event_shape is ().
        if loc.ndim > 0:
            batch_shape = loc.shape[:-1]
            event_shape = loc.shape[-1:]
        else:
            batch_shape = torch.Size([])
            event_shape = torch.Size([])

        self._event_shape = event_shape
        super().__init__(shape=batch_shape, device=loc.device)

    @classmethod
    def _unflatten_distribution(
        cls, attributes: Dict[str, Any]
    ) -> TensorTruncatedNormal:
        # Reconstruct event_shape from attributes if needed, or recompute
        # For now, recompute based on _loc
        loc = attributes["_loc"]
        if loc.ndim > 0:
            event_shape = loc.shape[-1:]
        else:
            event_shape = torch.Size([])

        instance = cls(
            loc=loc,
            scale=attributes["_scale"],
            low=attributes["_low"],
            high=attributes["_high"],
            validate_args=attributes.get("_validate_args"),
        )
        instance._event_shape = event_shape  # Set event_shape after init
        return instance

    @property
    def loc(self) -> Tensor:
        return self._loc

    @property
    def scale(self) -> Tensor:
        return self._scale

    @property
    def low(self) -> Tensor:
        return self._low

    @property
    def high(self) -> Tensor:
        return self._high

    @property
    def event_shape(self) -> Size:
        return self._event_shape

    def dist(self) -> Distribution:
        return Independent(
            TruncatedNormal(
                self.loc,
                self.scale,
                self.low,
                self.high,
                eps=1e-6,  # Explicitly pass eps
            ),
            len(self.event_shape),
        )
