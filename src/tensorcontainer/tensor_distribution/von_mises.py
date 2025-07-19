from __future__ import annotations

from typing import Any, Dict, Optional, cast

import torch
from torch.distributions import VonMises as TorchVonMises
from torch.distributions.utils import broadcast_all

from .base import TensorDistribution


class TensorVonMises(TensorDistribution):
    """Tensor-aware VonMises distribution."""

    # Annotated tensor parameters
    _loc: torch.Tensor
    _concentration: torch.Tensor
    validate_args: Optional[bool]

    def __init__(
        self,
        loc: torch.Tensor,
        concentration: torch.Tensor,
        validate_args: Optional[bool] = None,
    ):
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._loc, self._concentration = broadcast_all(loc, concentration)
        self.validate_args = validate_args
        super().__init__(self._loc.shape, self._loc.device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorVonMises:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=cast(torch.Tensor, attributes["_loc"]),
            concentration=cast(torch.Tensor, attributes["_concentration"]),
            validate_args=cast(Optional[bool], attributes.get("validate_args")),
        )

    def dist(self) -> TorchVonMises:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchVonMises(
            loc=self._loc,
            concentration=self._concentration,
            validate_args=self.validate_args,
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.dist().log_prob(value)

    @property
    def loc(self) -> torch.Tensor:
        """Returns the loc parameter of the distribution."""
        return self.dist().loc

    @property
    def concentration(self) -> torch.Tensor:
        """Returns the concentration parameter of the distribution."""
        return self.dist().concentration
