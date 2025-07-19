from __future__ import annotations

from typing import Any, Dict, Union

import torch
from torch import Tensor
from torch.distributions import InverseGamma as TorchInverseGamma

from .base import TensorDistribution


class TensorInverseGamma(TensorDistribution):
    """Tensor-aware Inverse Gamma distribution."""

    # Annotated tensor parameters
    _concentration: Tensor
    _rate: Tensor

    def __init__(self, concentration: Union[float, Tensor], rate: Union[float, Tensor]):
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._concentration = (
            concentration
            if isinstance(concentration, Tensor)
            else torch.tensor(concentration)
        )
        self._rate = rate if isinstance(rate, Tensor) else torch.tensor(rate)

        shape = self._concentration.shape
        device = self._concentration.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorInverseGamma:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration=attributes["_concentration"],
            rate=attributes["_rate"],
        )

    def dist(self) -> TorchInverseGamma:
        """
        Returns the underlying torch.distributions.InverseGamma instance.
        """
        return TorchInverseGamma(
            concentration=self._concentration,
            rate=self._rate,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def concentration(self) -> Tensor:
        """Returns the concentration used to initialize the distribution."""
        return self._concentration

    @property
    def rate(self) -> Tensor:
        """Returns the rate used to initialize the distribution."""
        return self._rate
