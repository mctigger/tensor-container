from __future__ import annotations

from typing import Any, Dict, Union

import torch
from torch import Tensor
from torch.distributions import Cauchy
from torch.distributions.utils import broadcast_all

from .base import TensorDistribution


class TensorCauchy(TensorDistribution):
    """Tensor-aware Cauchy distribution.

    Creates a Cauchy distribution parameterized by `loc` (location) and `scale` parameters.
    The Cauchy distribution is a continuous probability distribution with heavy tails.

    Args:
        loc (float or Tensor): mode or median of the distribution.
        scale (float or Tensor): half width at half maximum.

    Note:
        The Cauchy distribution has no finite mean or variance. These properties
        are not implemented as they would return undefined values.
    """

    # Annotated tensor parameters
    _loc: Union[Tensor, float]
    _scale: Union[Tensor, float]

    def __init__(self, loc: Union[Tensor, float], scale: Union[Tensor, float]):
        self._loc, self._scale = broadcast_all(loc, scale)

        if isinstance(loc, (float, int)) and isinstance(scale, (float, int)):
            batch_shape = torch.Size()
            device = None
        else:
            batch_shape = self._loc.size()
            device = self._loc.device

        super().__init__(batch_shape, device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorCauchy:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=attributes["_loc"],
            scale=attributes["_scale"],
        )

    def dist(self) -> Cauchy:
        return Cauchy(loc=self._loc, scale=self._scale)

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
        return self.dist().mean

    @property
    def mode(self) -> Tensor:
        return self.dist().mode

    @property
    def variance(self) -> Tensor:
        return self.dist().variance
