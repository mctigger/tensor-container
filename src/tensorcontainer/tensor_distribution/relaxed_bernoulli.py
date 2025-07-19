from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Tensor
from torch.distributions import RelaxedBernoulli as TorchRelaxedBernoulli

from .base import TensorDistribution


class TensorRelaxedBernoulli(TensorDistribution):
    """Tensor-aware RelaxedBernoulli distribution."""

    # Annotated tensor parameters
    _temperature: Tensor
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(
        self,
        temperature: Tensor,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
    ):
        self._temperature = temperature
        self._probs = probs
        self._logits = logits
        super().__init__(temperature.shape, temperature.device)

    @classmethod
    def _unflatten_distribution(
        cls, attributes: Dict[str, Any]
    ) -> "TensorRelaxedBernoulli":
        """Reconstruct distribution from tensor attributes."""
        return cls(
            temperature=attributes["_temperature"],  # type: ignore
            probs=attributes.get("_probs"),  # type: ignore
            logits=attributes.get("_logits"),  # type: ignore
        )

    def dist(self) -> TorchRelaxedBernoulli:
        return TorchRelaxedBernoulli(
            temperature=self._temperature,
            probs=self._probs,
            logits=self._logits,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def temperature(self) -> Tensor:
        """Returns the temperature used to initialize the distribution."""
        return self.dist().temperature

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs
