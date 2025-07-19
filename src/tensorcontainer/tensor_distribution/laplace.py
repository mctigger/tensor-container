from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Laplace

from .base import TensorDistribution


class TensorLaplace(TensorDistribution):
    """Tensor-aware Laplace distribution."""

    # Annotated tensor parameters
    _loc: Tensor
    _scale: Tensor

    def __init__(self, loc: Tensor | float, scale: Tensor | float):
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._loc = loc if isinstance(loc, Tensor) else torch.tensor(loc)
        self._scale = scale if isinstance(scale, Tensor) else torch.tensor(scale)

        if torch.any(self._scale <= 0):
            raise ValueError("scale must be positive")

        try:
            torch.broadcast_tensors(self._loc, self._scale)
        except RuntimeError as e:
            raise ValueError(f"loc and scale must have compatible shapes: {e}")

        shape = torch.broadcast_shapes(self._loc.shape, self._scale.shape)
        device = self._loc.device if self._loc.numel() > 0 else self._scale.device
        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorLaplace:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=attributes["_loc"],
            scale=attributes["_scale"],
        )

    def dist(self) -> Laplace:
        return Laplace(
            loc=self._loc,
            scale=self._scale,
        )

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
        """Returns the variance of the Laplace distribution."""
        assert self._scale is not None
        return self.dist().variance
