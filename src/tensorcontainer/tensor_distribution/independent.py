from __future__ import annotations

from typing import Any, Dict

from torch import Size
from torch.distributions import Independent

from tensorcontainer.tensor_distribution.base import TensorDistribution


class TensorIndependent(TensorDistribution):
    base_distribution: TensorDistribution
    reinterpreted_batch_ndims: int

    def __init__(
        self, base_distribution: TensorDistribution, reinterpreted_batch_ndims: int
    ):
        self.base_distribution = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

        super().__init__(
            Size(base_distribution.shape[:-reinterpreted_batch_ndims]),
            base_distribution.device,
        )

    def dist(self):
        return Independent(
            self.base_distribution.dist(), self.reinterpreted_batch_ndims
        )

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]):
        return cls(
            base_distribution=attributes["base_distribution"],
            reinterpreted_batch_ndims=attributes["reinterpreted_batch_ndims"],
        )
