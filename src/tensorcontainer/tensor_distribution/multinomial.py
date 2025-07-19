from __future__ import annotations

from typing import Any, Dict, Optional, Union

from torch import Size, Tensor
from torch.distributions import Multinomial

from .base import TensorDistribution


class TensorMultinomial(TensorDistribution):
    """Tensor-aware Multinomial distribution."""

    # Annotated tensor parameters
    _total_count: int
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(
        self,
        total_count: Union[int, Tensor] = 1,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
    ):
        if probs is None and logits is None:
            raise RuntimeError("Either 'probs' or 'logits' must be provided.")
        if probs is not None and logits is not None:
            raise RuntimeError("Only one of 'probs' or 'logits' can be provided.")

        data = probs if probs is not None else logits
        if (
            data is None
        ):  # This case is already handled by the above checks, but for mypy
            raise RuntimeError("Internal error: data tensor is None.")

        # Store the parameters in annotated attributes before calling super().__init__()
        if isinstance(total_count, Tensor):
            self._total_count = int(total_count.item())
        else:
            self._total_count = total_count
        self._probs = probs
        self._logits = logits

        # The batch shape is all dimensions except the last one.
        shape = data.shape[:-1]
        device = data.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorMultinomial:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            total_count=attributes["_total_count"],
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
        )

    def dist(self) -> Multinomial:  # Changed return type
        return Multinomial(  # Removed Independent wrapper
            total_count=self._total_count, probs=self._probs, logits=self._logits
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def total_count(self) -> int:
        """Returns the total_count used to initialize the distribution."""
        return self._total_count

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits  # Access directly

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs  # Access directly

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        # The param_shape should be the shape of the probs/logits tensor
        # including the last dimension (number of categories)
        if self._probs is not None:
            return self._probs.shape
        elif self._logits is not None:
            return self._logits.shape
        else:
            raise RuntimeError("Neither probs nor logits are available.")
