from __future__ import annotations

from typing import Any, Dict, Union

from torch import Tensor
from torch.distributions import Gamma as TorchGamma
from torch.distributions.utils import broadcast_all

from .base import TensorDistribution


class TensorGamma(TensorDistribution):
    """Tensor-aware Gamma distribution.

    Creates a Gamma distribution parameterized by shape `concentration` and `rate`.

    Args:
        concentration: shape parameter of the distribution (often referred to as alpha)
        rate: rate parameter of the distribution (often referred to as beta), rate = 1 / scale
    """

    _concentration: Tensor
    _rate: Tensor

    def __init__(
        self,
        concentration: Union[float, Tensor],
        rate: Union[float, Tensor],
    ):
        self._concentration, self._rate = broadcast_all(concentration, rate)
        batch_shape = self._concentration.shape
        super().__init__(batch_shape, self._concentration.device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorGamma:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration=attributes["_concentration"],
            rate=attributes["_rate"],
        )

    def dist(self) -> TorchGamma:
        """Return Gamma distribution."""
        return TorchGamma(
            concentration=self._concentration,
            rate=self._rate,
        )

    @property
    def concentration(self) -> Tensor:
        """Returns the concentration parameter of the distribution."""
        return self._concentration

    @property
    def rate(self) -> Tensor:
        """Returns the rate parameter of the distribution."""
        return self._rate

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

    @property
    def mode(self) -> Tensor:
        """Returns the mode of the distribution."""
        return self.dist().mode
