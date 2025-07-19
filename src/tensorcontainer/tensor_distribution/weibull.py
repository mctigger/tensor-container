from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Weibull as TorchWeibull

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorWeibull(TensorDistribution):
    """Tensor-aware Weibull distribution."""

    # Annotated tensor parameters
    _scale: Tensor
    _concentration: Tensor
    _validate_args: Optional[bool]

    def __init__(
        self,
        scale: TDCompatible,
        concentration: TDCompatible,
        validate_args: Optional[bool] = None,
    ):
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._scale = torch.as_tensor(scale)
        self._concentration = torch.as_tensor(concentration)
        self._validate_args = validate_args

        shape = self._scale.shape
        device = self._scale.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorWeibull:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            scale=attributes["_scale"],  # type: ignore
            concentration=attributes["_concentration"],  # type: ignore
            validate_args=attributes["_validate_args"],
        )

    def dist(self) -> TorchWeibull:
        return TorchWeibull(
            scale=self._scale,
            concentration=self._concentration,
            validate_args=self._validate_args,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def scale(self) -> Tensor:
        """Returns the scale used to initialize the distribution."""
        return self._scale

    @property
    def concentration(self) -> Tensor:
        """Returns the concentration used to initialize the distribution."""
        return self._concentration
