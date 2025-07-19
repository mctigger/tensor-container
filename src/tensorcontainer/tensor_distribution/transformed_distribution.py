from __future__ import annotations

from typing import List

from torch.distributions import Distribution
from torch.distributions import TransformedDistribution as TorchTransformedDistribution
from torch.distributions.transforms import Transform

from tensorcontainer.tensor_distribution.base import TensorDistribution


class TransformedDistribution(TensorDistribution):
    """
    Creates a transformed distribution.

    Args:
        base_distribution (TensorDistribution): The base distribution.
        transforms (List[Transform]): A list of transforms.
    """

    base_distribution: TensorDistribution
    transforms: List[Transform]

    def __init__(
        self, base_distribution: TensorDistribution, transforms: List[Transform]
    ):
        self.base_distribution = base_distribution
        self.transforms = transforms
        super().__init__(base_distribution.batch_shape, base_distribution.device)

    def dist(self) -> Distribution:
        """
        Returns the underlying torch.distributions.Distribution instance.
        """
        return TorchTransformedDistribution(
            base_distribution=self.base_distribution.dist(),
            transforms=self.transforms,
        )
