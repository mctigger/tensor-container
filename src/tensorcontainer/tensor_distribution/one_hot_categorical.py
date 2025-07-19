from __future__ import annotations

from typing import Any, Dict, Optional

from torch import Size, Tensor
from torch.distributions import OneHotCategorical as TorchOneHotCategorical

from .base import TensorDistribution


class TensorOneHotCategorical(TensorDistribution):
    """Tensor-aware OneHotCategorical distribution."""

    # Annotated tensor parameters
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(self, probs: Optional[Tensor] = None, logits: Optional[Tensor] = None):
        data = probs if probs is not None else logits
        # Parameter validation occurs in super().__init__(), but we need an early
        # check here to safely derive shape and device from the data tensor
        # before calling the parent constructor
        if data is None:
            raise RuntimeError("Either 'probs' or 'logits' must be provided.")

        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._probs = probs
        self._logits = logits

        # For categorical distributions the last dimension must not be part of the
        # shape since it contains the probabilities for each class and thus, should
        # never change.
        shape = data.shape[:-1]
        device = data.device

        # The batch shape is all dimensions except the last one.
        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorOneHotCategorical:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            probs=attributes.get("_probs", None),  # type: ignore
            logits=attributes.get("_logits", None),  # type: ignore
        )

    def dist(self) -> TorchOneHotCategorical:
        return TorchOneHotCategorical(probs=self._probs, logits=self._logits)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.dist().param_shape
