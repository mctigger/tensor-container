from __future__ import annotations

from typing import Any, Dict, get_args

from torch import Tensor
from torch.distributions import Normal
from torch.distributions.utils import broadcast_all
from torch.types import Number

from .base import TensorDistribution


class TensorNormal(TensorDistribution):
    """Tensor-aware Normal distribution.

    Creates a Normal distribution parameterized by `loc` (mean) and `scale` (standard deviation).

    Args:
        loc: Mean of the distribution.
        scale: Standard deviation of the distribution. Must be positive.

    Note:
        The Normal distribution is also known as the Gaussian distribution.
    """

    # Annotated tensor parameters
    _loc: Tensor
    _scale: Tensor

    def __init__(self, loc: Tensor, scale: Tensor):
        self._loc, self._scale = broadcast_all(loc, scale)

        if isinstance(loc, get_args(Number)) and isinstance(scale, get_args(Number)):
            shape = tuple()
        else:
            shape = self._loc.shape

        device = self._loc.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, Any],
    ) -> TensorNormal:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=attributes.get("_loc"),  # type: ignore
            scale=attributes.get("_scale"),  # type: ignore
        )

    def dist(self) -> Normal:
        """Return Normal distribution."""
        return Normal(
            loc=self._loc,
            scale=self._scale,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def loc(self) -> Tensor:
        """Returns the location parameter of the distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the distribution."""
        return self.dist().scale

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
