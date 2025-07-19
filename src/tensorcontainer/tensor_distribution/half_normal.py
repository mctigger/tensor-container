from __future__ import annotations

from typing import Any, Dict, Optional, Union, cast

import torch
from torch import Tensor
from torch.distributions import HalfNormal

from .base import TensorDistribution


class TensorHalfNormal(TensorDistribution):
    """Tensor-aware HalfNormal distribution."""

    # Annotated tensor parameters
    _scale: Union[Tensor, float]
    _validate_args: Optional[bool] = None

    def __init__(
        self, scale: Union[Tensor, float], validate_args: Optional[bool] = None
    ):
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._scale = scale
        self._validate_args = validate_args

        if isinstance(scale, Tensor):
            shape = scale.shape
            device = scale.device
        else:
            shape = torch.Size([])
            device = None

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorHalfNormal:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            scale=cast(Union[Tensor, float], attributes.get("_scale")),
            validate_args=cast(Optional[bool], attributes.get("_validate_args")),
        )

    def dist(self) -> HalfNormal:
        return HalfNormal(scale=self._scale, validate_args=self._validate_args)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the HalfNormal distribution."""
        return self.dist().variance

    @property
    def scale(self) -> Tensor:
        """Returns the scale used to initialize the distribution."""
        return self.dist().scale
