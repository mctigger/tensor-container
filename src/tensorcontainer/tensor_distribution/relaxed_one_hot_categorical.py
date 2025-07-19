from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from torch import Tensor
from torch.distributions import RelaxedOneHotCategorical

from .base import TensorDistribution


class TensorRelaxedOneHotCategorical(TensorDistribution):
    """Tensor-aware RelaxedCategorical distribution."""

    # Annotated tensor parameters
    _temperature: Tuple[Tensor]
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None
    _validate_args: Optional[bool] = None

    def __init__(
        self,
        temperature: Tensor,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args: Optional[bool] = None,
    ):
        """
        There is a bug in RelaxedOneHotCategorical https://github.com/pytorch/pytorch/issues/37162
        That is why we only allowed scalar temperatures for now.
        """
        if temperature.ndim > 0:
            raise RuntimeError(
                "Expected scalar temperature tensor. This is because of a bug in torch: https://github.com/pytorch/pytorch/issues/37162"
            )

        data = probs if probs is not None else logits
        if data is None:
            raise RuntimeError("Either 'probs' or 'logits' must be provided.")

        # Determine shape and device from data (probs or logits)
        shape = data.shape[:-1]
        device = data.device

        self._temperature = (temperature,)
        self._probs = probs
        self._logits = logits
        self._validate_args = validate_args

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> "TensorRelaxedOneHotCategorical":
        """Reconstruct distribution from tensor attributes."""
        return cls(
            temperature=attributes["_temperature"][0],  # type: ignore
            probs=attributes.get("_probs"),  # type: ignore
            logits=attributes.get("_logits"),  # type: ignore
            validate_args=attributes.get("_validate_args"),  # type: ignore
        )

    def dist(self) -> RelaxedOneHotCategorical:
        return RelaxedOneHotCategorical(
            temperature=self._temperature[0],
            probs=self._probs,
            logits=self._logits,
            validate_args=self._validate_args,
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
