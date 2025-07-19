from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import FisherSnedecor as TorchFisherSnedecor
from torch.distributions.utils import broadcast_all

from .base import TensorDistribution


class TensorFisherSnedecor(TensorDistribution):
    def __init__(
        self, df1: Tensor, df2: Tensor, *, validate_args: Optional[bool] = None
    ):
        self._df1: Tensor
        self._df2: Tensor
        self._df1, self._df2 = broadcast_all(df1, df2)
        batch_shape = self._df1.shape
        super().__init__(batch_shape, self._df1.device)

    def dist(self) -> TorchFisherSnedecor:
        return TorchFisherSnedecor(self._df1, self._df2)

    @classmethod
    def _unflatten_distribution(
        cls, attributes: Dict[str, Any]
    ) -> "TensorFisherSnedecor":
        return cls(
            df1=attributes["_df1"],
            df2=attributes["_df2"],
            validate_args=attributes.get("validate_args"),
        )

    @property
    def mean(self) -> Tensor:
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        return self.dist().variance

    @property
    def mode(self) -> Tensor:
        return self.dist().mode

    @property
    def support(self):
        return self.dist().support

    @property
    def has_rsample(self):
        return self.dist().has_rsample

    def entropy(self) -> Tensor:
        return self.dist().entropy()

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    def cdf(self, value: Tensor) -> Tensor:
        return self.dist().cdf(value)

    def icdf(self, value: Tensor) -> Tensor:
        return self.dist().icdf(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self.dist().sample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        return self.dist().rsample(sample_shape)

    def enumerate_support(self, expand: bool = True) -> Tensor:
        return self.dist().enumerate_support(expand)

    def __repr__(self):
        return self.dist().__repr__()

    def __eq__(self, other):
        return self.dist().__eq__(other)
