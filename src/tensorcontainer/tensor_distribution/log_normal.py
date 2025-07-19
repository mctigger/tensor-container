from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Size, Tensor
from torch.distributions import LogNormal

from .base import TensorDistribution


class TensorLogNormal(TensorDistribution):
    """Tensor-aware LogNormal distribution."""

    # Annotated tensor parameters
    _loc: Tensor
    _scale: Tensor

    def __init__(self, loc: float | Tensor, scale: float | Tensor):
        # Convert inputs to tensors
        loc = torch.as_tensor(loc)
        scale = torch.as_tensor(scale)

        try:
            data = torch.broadcast_tensors(loc, scale)
        except RuntimeError as e:
            raise ValueError(f"loc and scale must have compatible shapes: {e}")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._loc = data[0]
        self._scale = data[1]

        if torch.any(self._scale <= 0):
            raise ValueError("scale must be positive")

        shape = self._loc.shape
        device = self._loc.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorLogNormal:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=attributes.get("_loc"),  # type: ignore
            scale=attributes.get("_scale"),  # type: ignore
        )

    def dist(self) -> LogNormal:
        return LogNormal(loc=self._loc, scale=self._scale)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def loc(self) -> Optional[Tensor]:
        """Returns the loc used to initialize the distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Optional[Tensor]:
        """Returns the scale used to initialize the distribution."""
        return self.dist().scale

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the LogNormal distribution."""
        return self.dist().variance

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.batch_shape
