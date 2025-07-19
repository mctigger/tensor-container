from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor
from torch.distributions import Dirichlet

from .base import TensorDistribution


class TensorDirichlet(TensorDistribution):
    """Tensor-aware Dirichlet distribution."""

    # Annotated tensor parameters
    _concentration: Tensor

    def __init__(self, concentration: Tensor, validate_args: Optional[bool] = None):
        self._concentration = concentration
        batch_shape = concentration.shape[:-1]
        super().__init__(batch_shape, concentration.device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorDirichlet:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration=attributes["_concentration"],
            validate_args=attributes.get("validate_args"),
        )

    def dist(self) -> Dirichlet:
        return Dirichlet(concentration=self._concentration)

    @property
    def concentration(self) -> Tensor:
        return self.dist().concentration

    @property
    def mean(self) -> Tensor:
        return self.dist().mean

    @property
    def mode(self) -> Tensor:
        return self.dist().mode

    @property
    def variance(self) -> Tensor:
        return self.dist().variance
