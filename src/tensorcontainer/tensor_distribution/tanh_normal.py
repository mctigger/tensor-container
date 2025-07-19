from __future__ import annotations

from typing import Any, Dict, Optional, get_args

import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    TransformedDistribution,
    constraints,
)
from torch.distributions.utils import broadcast_all
from torch.types import Number

from .base import TensorDistribution


class ClampedTanhTransform(torch.distributions.transforms.Transform):
    """
    Transform that applies tanh and clamps the output between -1 and 1.
    """

    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True

    @property
    def sign(self):
        return +1

    def __init__(self):
        super().__init__()

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        # Arctanh
        return torch.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # |det J| = 1 - tanh^2(x)
        # log|det J| = log(1 - tanh^2(x))
        return torch.log(
            1 - y.pow(2) + 1e-6
        )  # Adding small epsilon for numerical stability


class TensorTanhNormal(TensorDistribution):
    _loc: Tensor
    _scale: Tensor
    _reinterpreted_batch_ndims: int

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        reinterpreted_batch_ndims: Optional[int] = None,
    ):
        self._loc, self._scale = broadcast_all(loc, scale)

        if reinterpreted_batch_ndims is None:
            self._reinterpreted_batch_ndims = 0
            if self._loc.ndim > 0:
                self._reinterpreted_batch_ndims = 1
        else:
            self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

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
    ) -> TensorTanhNormal:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            loc=attributes.get("_loc"),  # type: ignore
            scale=attributes.get("_scale"),  # type: ignore
            reinterpreted_batch_ndims=attributes.get("_reinterpreted_batch_ndims"),  # type: ignore
        )

    def dist(self) -> Distribution:
        return Independent(
            TransformedDistribution(
                Normal(self._loc.float(), self._scale.float(), validate_args=False),
                [
                    ClampedTanhTransform(),
                ],
                validate_args=False,
            ),
            self._reinterpreted_batch_ndims,
        )

    @property
    def loc(self) -> Tensor:
        return self._loc

    @property
    def scale(self) -> Tensor:
        return self._scale
