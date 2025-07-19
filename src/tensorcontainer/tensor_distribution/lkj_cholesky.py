from __future__ import annotations

from typing import Any, Dict, Union

import torch
from torch import Tensor
from torch.distributions import LKJCholesky as TorchLKJCholesky
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution


class TensorLKJCholesky(TensorDistribution):
    """
    Creates a distribution of Cholesky factors of correlation matrices.

    The distribution is defined over the space of `d x d` lower-triangular
    matrices `L` with positive diagonal entries, such that `L @ L.T` is a
    correlation matrix.

    Args:
        dim (int): The dimension of the correlation matrix.
        concentration (float or Tensor): The concentration parameter of the distribution.
            Must be positive.
    """

    _dim: int
    _concentration: Tensor

    def __init__(
        self,
        dim: int,
        concentration: Union[float, Tensor] = 1.0,
    ):
        self._dim = dim
        self._concentration = (
            concentration
            if isinstance(concentration, Tensor)
            else torch.tensor(concentration, dtype=torch.float)
        )
        super().__init__(
            shape=self._concentration.shape,
            device=self._concentration.device,
        )

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchLKJCholesky(
            dim=self._dim,
            concentration=self._concentration,
        )

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorLKJCholesky:
        return cls(
            dim=attributes["_dim"],
            concentration=attributes["_concentration"],
        )
