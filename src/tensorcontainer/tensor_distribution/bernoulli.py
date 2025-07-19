from __future__ import annotations

from typing import Any, Dict, Optional, Union

from torch import Size, Tensor
from torch.distributions import Bernoulli
from torch.distributions.utils import broadcast_all
from torch.types import Number

from .base import TensorDistribution


class TensorBernoulli(TensorDistribution):
    """Tensor-aware Bernoulli distribution."""

    # Annotated tensor parameters
    _probs: Optional[Tensor]
    _logits: Optional[Tensor]

    def __init__(
        self,
        probs: Optional[Union[Number, Tensor]] = None,
        logits: Optional[Union[Number, Tensor]] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        if probs is not None:
            (self._probs,) = broadcast_all(probs)
            shape = self._probs.shape
            device = self._probs.device
            self._logits = None
        else:
            (self._logits,) = broadcast_all(logits)
            shape = self._logits.shape
            device = self._logits.device
            self._probs = None

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorBernoulli:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
        )

    def dist(self) -> Bernoulli:
        return Bernoulli(probs=self._probs, logits=self._logits)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def mean(self) -> Tensor:
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        return self.dist().variance

    @property
    def mode(self) -> Tensor:
        return self.dist().mode

    @property
    def logits(self) -> Tensor:
        return self.dist().logits

    @property
    def probs(self) -> Tensor:
        return self.dist().probs

    @property
    def param_shape(self) -> Size:
        return self.dist().param_shape
