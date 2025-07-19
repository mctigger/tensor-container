from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Poisson

from .base import TensorDistribution


class TensorPoisson(TensorDistribution):
    # Annotated tensor parameters
    _rate: Optional[Tensor] = None

    def __init__(self, rate: torch.Tensor, validate_args: Optional[bool] = None):
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._rate = rate

        shape = rate.shape
        device = rate.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorPoisson:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            rate=attributes.get("_rate"),  # type: ignore
        )

    def dist(self) -> Poisson:
        return Poisson(rate=self._rate)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def rate(self) -> Optional[Tensor]:
        """Returns the rate parameter used to initialize the distribution."""
        return self.dist().rate
