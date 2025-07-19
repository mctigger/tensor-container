from __future__ import annotations

from typing import Any, Dict, Union

from torch import Tensor
from torch.distributions import Pareto as TorchPareto

from .base import TensorDistribution


class TensorPareto(TensorDistribution):
    """
    A Pareto distribution.

    This distribution is parameterized by `scale` and `alpha`.

    Source: https://pytorch.org/docs/stable/distributions.html#pareto
    """

    # Annotated tensor parameters
    _scale: Tensor
    _alpha: Tensor

    def __init__(
        self, scale: Union[float, Tensor], alpha: Union[float, Tensor]
    ) -> None:
        if isinstance(scale, (float, int)):
            scale = Tensor([scale])
        if isinstance(alpha, (float, int)):
            alpha = Tensor([alpha])

        if scale is None:
            raise RuntimeError("`scale` must be provided.")
        if alpha is None:
            raise RuntimeError("`alpha` must be provided.")

        self._scale = scale
        self._alpha = alpha

        shape = self._scale.shape
        device = self._scale.device

        super().__init__(shape, device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]) -> TensorPareto:
        """Reconstruct distribution from tensor attributes."""
        return cls(
            scale=attributes["_scale"],  # type: ignore
            alpha=attributes["_alpha"],  # type: ignore
        )

    def dist(self) -> TorchPareto:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchPareto(
            scale=self._scale,
            alpha=self._alpha,
        )

    def log_prob(self, value: Tensor) -> Tensor:
        return self.dist().log_prob(value)

    @property
    def scale(self) -> Tensor:
        """Returns the scale parameter of the distribution."""
        return self.dist().scale

    @property
    def alpha(self) -> Tensor:
        """Returns the alpha parameter of the distribution."""
        return self.dist().alpha
