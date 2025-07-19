from __future__ import annotations

from typing import Any, Dict

from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily
from torch.distributions.distribution import Distribution

from tensorcontainer.tensor_distribution.base import TensorDistribution
from tensorcontainer.tensor_distribution.categorical import TensorCategorical


class TensorMixtureSameFamily(TensorDistribution):
    """
    Creates a mixture of distributions of the same family.

    Args:
        mixture_distribution (Categorical): The mixture distribution.
        component_distribution (TensorDistribution): The component distribution.
    """

    _mixture_distribution: TensorCategorical
    _component_distribution: TensorDistribution

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchMixtureSameFamily(
            mixture_distribution=self._mixture_distribution.dist(),
            component_distribution=self._component_distribution.dist(),
        )

    def __init__(
        self,
        mixture_distribution: TensorCategorical,
        component_distribution: TensorDistribution,
    ) -> None:
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution
        super().__init__(
            shape=mixture_distribution.batch_shape,
            device=mixture_distribution.device,
        )

    @property
    def mixture_distribution(self) -> TensorCategorical:
        return self._mixture_distribution

    @property
    def component_distribution(self) -> TensorDistribution:
        return self._component_distribution

    @classmethod
    def _unflatten_distribution(
        cls, attributes: Dict[str, Any]
    ) -> TensorMixtureSameFamily:
        return cls(
            mixture_distribution=attributes["_mixture_distribution"],
            component_distribution=attributes["_component_distribution"],
        )
