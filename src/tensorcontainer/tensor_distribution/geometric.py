from __future__ import annotations

from numbers import Number
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Geometric

from .base import TensorDistribution


class TensorGeometric(TensorDistribution):
    """Tensor-aware Geometric distribution."""

    # Annotated tensor parameters
    _probs: Optional[Union[Tensor, Number]] = None
    _logits: Optional[Union[Tensor, Number]] = None

    def __init__(
        self,
        probs: Optional[Union[Tensor, Number]] = None,
        logits: Optional[Union[Tensor, Number]] = None,
    ):
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        if probs is not None:
            self._probs = probs
        else:
            self._logits = logits

        # Create a temporary distribution to get the batch_shape and device
        # Create a temporary distribution to get the batch_shape and device
        temp_dist = Geometric(probs=probs, logits=logits)

        # Determine the device based on which parameter is used
        if logits is not None:
            device = (
                logits.device if isinstance(logits, Tensor) else torch.device("cpu")
            )
        else:
            device = probs.device if isinstance(probs, Tensor) else torch.device("cpu")

        super().__init__(temp_dist.batch_shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorGeometric:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
        )

    def dist(self) -> Geometric:
        return Geometric(probs=self._probs, logits=self._logits)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def probs(self) -> Optional[Tensor]:
        """Returns the probabilities used to initialize the distribution."""
        return self.dist().probs

    @property
    def logits(self) -> Optional[Tensor]:
        """Returns the logits used to initialize the distribution."""
        return self.dist().logits

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the distribution."""
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the distribution."""
        return self.dist().variance

    @property
    def stddev(self) -> Tensor:
        """Returns the standard deviation of the distribution."""
        return self.dist().stddev

    def entropy(self) -> Tensor:
        """Returns the entropy of the distribution."""
        return self.dist().entropy()

    @property
    def support(self) -> Any:
        """Returns the support of the distribution."""
        return self.dist().support

    @property
    def arg_constraints(self) -> Dict[str, Any]:
        """Returns the argument constraints of the distribution."""
        return self.dist().arg_constraints

    @property
    def batch_shape(self) -> torch.Size:
        """Returns the batch shape of the distribution."""
        return self.dist().batch_shape

    @property
    def event_shape(self) -> torch.Size:
        """Returns the event shape of the distribution."""
        return self.dist().event_shape

    @property
    def has_rsample(self) -> bool:
        """Returns True if the distribution has a reparameterization trick."""
        return self.dist().has_rsample

    @property
    def has_enumerate_support(self) -> bool:
        """Returns True if the distribution has enumerate_support implemented."""
        return self.dist().has_enumerate_support

    @property
    def _validate_args(self) -> bool:
        """Returns True if the distribution validates arguments."""
        return self.dist()._validate_args
