from __future__ import annotations

from typing import Dict, Optional, Union

import torch
from torch import Tensor
from torch.distributions import StudentT

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorStudentT(TensorDistribution):
    """Tensor-aware StudentT distribution."""

    # Annotated tensor parameters
    _df: Optional[Tensor] = None
    _loc: Optional[Tensor] = None
    _scale: Optional[Tensor] = None

    def __init__(
        self,
        df: Union[float, Tensor],
        loc: Union[float, Tensor] = 0.0,
        scale: Union[float, Tensor] = 1.0,
        validate_args: Optional[bool] = None,
    ):
        # Convert to tensors and ensure compatible shapes
        if isinstance(df, (float, int)):
            df = torch.as_tensor(df)
        if isinstance(loc, (float, int)):
            loc = torch.as_tensor(loc, dtype=df.dtype, device=df.device)
        if isinstance(scale, (float, int)):
            scale = torch.as_tensor(scale, dtype=df.dtype, device=df.device)

        # Determine the common batch_shape
        try:
            batch_shape = torch.broadcast_shapes(df.shape, loc.shape, scale.shape)
        except RuntimeError as e:
            raise ValueError(f"df, loc, and scale must have compatible shapes: {e}")

        # Expand parameters to the common batch_shape
        self._df = df.expand(batch_shape)
        self._loc = loc.expand(batch_shape)
        self._scale = scale.expand(batch_shape)

        if torch.any(self._df <= 0):
            raise ValueError("df must be positive")
        if torch.any(self._scale <= 0):
            raise ValueError("scale must be positive")

        super().__init__(batch_shape, self._df.device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, TDCompatible],
    ) -> "TensorStudentT":
        """Reconstruct distribution from tensor attributes."""
        return cls(
            df=torch.as_tensor(attributes["_df"]),
            loc=torch.as_tensor(attributes["_loc"]),
            scale=torch.as_tensor(attributes["_scale"]),
        )

    def dist(self) -> StudentT:
        assert self._df is not None
        assert self._loc is not None
        assert self._scale is not None
        return StudentT(df=self._df, loc=self._loc, scale=self._scale)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def df(self) -> Optional[Tensor]:
        """Returns the degrees of freedom of the StudentT distribution."""
        return self.dist().df

    @property
    def loc(self) -> Optional[Tensor]:
        """Returns the mean of the StudentT distribution."""
        return self.dist().loc

    @property
    def scale(self) -> Optional[Tensor]:
        """Returns the scale of the StudentT distribution."""
        return self.dist().scale

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the StudentT distribution."""
        assert self._df is not None
        assert self._scale is not None
        var = torch.full_like(self._df, float("inf"))
        var[self._df > 2] = (self._scale**2 * (self._df / (self._df - 2)))[self._df > 2]
        var[self._df <= 1] = float("nan")
        return var
