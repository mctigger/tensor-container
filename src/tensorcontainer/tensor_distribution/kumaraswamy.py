from __future__ import annotations

from typing import Dict, Union

import torch
from torch import Tensor
from torch.distributions import Kumaraswamy as TorchKumaraswamy

from tensorcontainer.tensor_annotated import TDCompatible

from .base import TensorDistribution


class TensorKumaraswamy(TensorDistribution):
    """Tensor-aware Kumaraswamy distribution."""

    # Annotated tensor parameters
    _concentration1: Tensor
    _concentration0: Tensor

    def __init__(
        self, concentration1: Union[float, Tensor], concentration0: Union[float, Tensor]
    ):
        # Store the parameters in annotated attributes before calling super().__init__()
        # This is required because super().__init__() calls self.dist() which needs these attributes
        self._concentration1 = (
            concentration1
            if isinstance(concentration1, Tensor)
            else torch.tensor(concentration1)
        )
        self._concentration0 = (
            concentration0
            if isinstance(concentration0, Tensor)
            else torch.tensor(concentration0)
        )

        shape = self._concentration1.shape
        device = self._concentration1.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: Dict[str, TDCompatible],
    ) -> "TensorKumaraswamy":
        """Reconstruct distribution from tensor attributes."""
        return cls(
            concentration1=attributes["_concentration1"],  # type: ignore
            concentration0=attributes["_concentration0"],  # type: ignore
        )

    def dist(self) -> TorchKumaraswamy:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchKumaraswamy(
            concentration1=self._concentration1,
            concentration0=self._concentration0,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def concentration1(self) -> Tensor:
        """Returns the concentration1 parameter of the distribution."""
        return self.dist().concentration1

    @property
    def concentration0(self) -> Tensor:
        """Returns the concentration0 parameter of the distribution."""
        return self.dist().concentration0
