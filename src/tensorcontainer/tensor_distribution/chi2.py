from typing import Union

import torch
from torch import Tensor
from torch.distributions import Chi2 as TorchChi2

from tensorcontainer.tensor_distribution.base import TensorDistribution


class TensorChi2(TensorDistribution):
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter :attr:`df`.
    This is exactly equivalent to ``Gamma(alpha=0.5*df, beta=0.5)``

    Args:
        df (float or Tensor): shape parameter of the distribution
    """

    _df: Tensor

    def __init__(self, df: Union[float, Tensor]):
        if isinstance(df, (float, int)):  # Handle both float and int
            df = torch.tensor(df, dtype=torch.float32)

        self._df = df

        # Determine batch_shape and device from the (potentially broadcasted) parameters
        batch_shape = self._df.shape
        device = self._df.device

        super().__init__(shape=batch_shape, device=device)

    def dist(self) -> TorchChi2:
        return TorchChi2(df=self._df)

    @classmethod
    def _unflatten_distribution(cls, attributes: dict) -> "TensorChi2":
        return cls(df=attributes["_df"])

    @property
    def df(self) -> Tensor:
        return self.dist().df

    @property
    def mean(self) -> Tensor:
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        return self.dist().variance
