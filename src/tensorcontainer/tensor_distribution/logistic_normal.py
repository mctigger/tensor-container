from __future__ import annotations

from typing import Any, Dict, get_args

import torch  # Added import torch
from torch import Tensor
from torch.distributions import LogisticNormal as TorchLogisticNormal
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import Number

from .base import TensorDistribution


class TensorLogisticNormal(TensorDistribution):
    _loc: Tensor
    _scale: Tensor

    def __init__(self, loc: Tensor, scale: Tensor):
        self._loc, self._scale = broadcast_all(loc, scale)

        if isinstance(loc, get_args(Number)) and isinstance(scale, get_args(Number)):
            shape = torch.Size([1])
            self._loc = self._loc.unsqueeze(0)  # Unsqueeze for scalar inputs
            self._scale = self._scale.unsqueeze(0)  # Unsqueeze for scalar inputs
        else:
            shape = self._loc.shape

        device = self._loc.device

        super().__init__(shape, device)

    def dist(self) -> Distribution:
        return TorchLogisticNormal(loc=self._loc, scale=self._scale)

    @property
    def loc(self) -> Tensor:
        """Returns the location parameter of the distribution."""
        return self._loc

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the distribution."""
        return self._scale

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorLogisticNormal:
        loc = attributes.get("_loc")
        scale = attributes.get("_scale")
        assert loc is not None and isinstance(loc, Tensor)
        assert scale is not None and isinstance(scale, Tensor)
        return cls(
            loc=loc,
            scale=scale,
        )
