from __future__ import annotations

from typing import Any, Dict, get_args

from torch import Tensor
from torch.distributions import Beta
from torch.distributions.utils import broadcast_all
from torch.types import Number

from .base import TensorDistribution


class TensorBeta(TensorDistribution):
    """Tensor-aware Beta distribution.

    Creates a Beta distribution parameterized by `concentration1` and `concentration0`.

    Args:
        concentration1: First concentration parameter (alpha). Must be positive.
        concentration0: Second concentration parameter (beta). Must be positive.

    Note:
        The Beta distribution is defined on the interval (0, 1) and is commonly used
        to model probabilities and proportions.
    """

    # Annotated tensor parameters
    _concentration1: Tensor
    _concentration0: Tensor

    def __init__(self, concentration1: Tensor, concentration0: Tensor):
        self._concentration1, self._concentration0 = broadcast_all(
            concentration1, concentration0
        )

        if isinstance(concentration1, get_args(Number)) and isinstance(
            concentration0, get_args(Number)
        ):
            shape = tuple()
        else:
            shape = self._concentration1.shape

        device = self._concentration1.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorBeta:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration1=attributes.get("_concentration1"),  # type: ignore
            concentration0=attributes.get("_concentration0"),  # type: ignore
        )

    def dist(self) -> Beta:
        """Return Beta distribution."""
        return Beta(
            concentration1=self._concentration1,
            concentration0=self._concentration0,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def concentration1(self) -> Tensor:
        """Returns the concentration1 parameter of the distribution."""
        return self.dist().concentration1

    @property
    def concentration0(self) -> Tensor:
        """Returns the concentration0 parameter of the distribution."""
        return self.dist().concentration0

    @property
    def mean(self) -> Tensor:
        """Returns the mean of the distribution."""
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the distribution."""
        return self.dist().variance

    @property
    def mode(self) -> Tensor:
        """Returns the mode of the distribution."""
        return self.dist().mode

    @property
    def stddev(self) -> Tensor:
        """Returns the standard deviation of the distribution."""
        return self.dist().stddev
