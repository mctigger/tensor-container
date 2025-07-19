from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from torch import Size, Tensor
from torch.distributions import Binomial
from torch.distributions.utils import broadcast_all

from .base import TensorDistribution


class TensorBinomial(TensorDistribution):
    """Tensor-aware Binomial distribution.

    Creates a Binomial distribution parameterized by `total_count` and either `probs`
    or `logits` (but not both). The distribution represents the number of successes
    in `total_count` independent Bernoulli trials.

    Args:
        total_count: Number of Bernoulli trials. Can be an int or Tensor.
        probs: Event probabilities. Must be in range [0, 1]. Mutually exclusive with logits.
        logits: Event log-odds (log(p/(1-p))). Mutually exclusive with probs.
    """

    # Annotated tensor parameters
    _total_count: Union[int, Tensor]
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None

    def __init__(
        self,
        total_count: Union[int, Tensor] = 1,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        if probs is not None:
            self._total_count, self._probs = broadcast_all(total_count, probs)
            param = self._probs
            assert param is not None
        else:
            self._total_count, self._logits = broadcast_all(total_count, logits)
            param = self._logits
            assert param is not None

        # Ensure total_count has the same dtype as the parameter tensor if it's a Tensor
        if isinstance(self._total_count, Tensor):
            self._total_count = self._total_count.type_as(param)

        shape = param.shape
        device = param.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]):
        """Reconstruct distribution from tensor attributes."""
        total_count = attributes["_total_count"]
        if isinstance(total_count, Tensor):
            total_count = total_count.clone()

        probs = attributes.get("_probs")
        if probs is not None:
            probs = probs.clone()

        logits = attributes.get("_logits")
        if logits is not None:
            logits = logits.clone()

        return cls(
            total_count=total_count,
            probs=probs,
            logits=logits,
        )

    def dist(self) -> Binomial:
        total_count = self._total_count
        if isinstance(total_count, int):
            # Convert int total_count to a tensor with the correct device and dtype
            # The device and dtype should match the probs/logits tensor
            if self._probs is not None:
                total_count = torch.tensor(
                    total_count, device=self._probs.device, dtype=self._probs.dtype
                )
            elif self._logits is not None:
                total_count = torch.tensor(
                    total_count, device=self._logits.device, dtype=self._logits.dtype
                )
            else:
                # Fallback if neither probs nor logits are set (should not happen with current init logic)
                total_count = torch.tensor(total_count)

        return Binomial(total_count=total_count, probs=self._probs, logits=self._logits)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def total_count(self) -> Union[int, Tensor]:
        """Returns the total_count parameter of the distribution."""
        return self._total_count

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probs parameter of the distribution."""
        return self.dist().probs

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits parameter of the distribution."""
        return self.dist().logits

    @property
    def param_shape(self) -> Size:
        """Returns the shape of the underlying parameter."""
        return self.dist().param_shape

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the Binomial distribution."""
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the Binomial distribution."""
        return self.dist().variance

    @property
    def mode(self) -> Tensor:
        """Returns the mode of the Binomial distribution."""
        return self.dist().mode
