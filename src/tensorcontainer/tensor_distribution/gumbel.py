from __future__ import annotations

from typing import Any, Dict, Union

import torch
from torch import Tensor
from torch.distributions import Gumbel as TorchGumbel

from tensorcontainer.tensor_container import TensorContainer

from .base import TensorDistribution


class TensorGumbel(TensorDistribution):
    """
    A Gumbel distribution.

    This distribution is parameterized by `loc` and `scale`.

    Source: https://pytorch.org/docs/stable/distributions.html#gumbel
    """

    _loc: Union[Tensor, float]
    _scale: Union[Tensor, float]

    def __init__(self, loc: Union[Tensor, float], scale: Union[Tensor, float]):
        self._loc = loc
        self._scale = scale
        if isinstance(loc, Tensor):
            shape = loc.shape
            device = loc.device
        elif isinstance(scale, Tensor):
            shape = scale.shape
            device = scale.device
        else:
            # If both are floats, assume scalar distribution on CPU
            shape = torch.Size([])
            device = torch.device("cpu")
        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorGumbel:
        """Reconstruct distribution from tensor attributes."""
        loc = attributes["_loc"]
        scale = attributes["_scale"]
        if isinstance(loc, TensorContainer):
            loc = loc.as_tensor()
        if isinstance(scale, TensorContainer):
            scale = scale.as_tensor()
        return cls(
            loc=loc,
            scale=scale,
        )

    def dist(self) -> TorchGumbel:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchGumbel(
            loc=self._loc,
            scale=self._scale,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def loc(self) -> Union[Tensor, float]:
        """Returns the loc used to initialize the distribution."""
        return self._loc

    @property
    def scale(self) -> Union[Tensor, float]:
        """Returns the scale used to initialize the distribution."""
        return self._scale
