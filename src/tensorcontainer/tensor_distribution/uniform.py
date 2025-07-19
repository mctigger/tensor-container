from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Uniform

from .base import TensorDistribution


class TensorUniform(TensorDistribution):
    _low: Optional[Tensor] = None
    _high: Optional[Tensor] = None

    def __init__(self, low: Tensor, high: Tensor, validate_args: Optional[bool] = None):
        low, high = torch.broadcast_tensors(low, high)
        self._low = low
        self._high = high
        self.validate_args = validate_args
        super().__init__(low.shape, low.device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorUniform:
        return cls(
            low=attributes.get("_low"),  # type: ignore
            high=attributes.get("_high"),  # type: ignore
            validate_args=attributes.get("validate_args"),
        )

    def dist(self) -> Uniform:
        return Uniform(low=self._low, high=self._high, validate_args=self.validate_args)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def low(self) -> Tensor:
        return self.dist().low

    @property
    def high(self) -> Tensor:
        return self.dist().high

    @property
    def mean(self) -> Tensor:
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        return self.dist().variance

    @property
    def stddev(self) -> Tensor:
        return self.dist().stddev
