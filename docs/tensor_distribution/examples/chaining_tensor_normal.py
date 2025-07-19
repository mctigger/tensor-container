import torch

from tensorcontainer.tensor_distribution import (
    TensorBernoulli,
    TensorNormal,
)
from tensorcontainer.tensor_distribution.base import TensorDistribution


def chain(distribution: TensorDistribution):
    distribution = distribution.view(2, 3, 4)
    distribution = distribution.permute(1, 0, 2)
    distribution = distribution.detach()

    return distribution


# Create a TensorNormal
loc = torch.randn(2 * 3 * 4)
scale = torch.abs(torch.randn(2 * 3 * 4))
normal = TensorNormal(loc=loc, scale=scale)

# Execute the chain for TensorNormal
chain(normal)

# Execute the chain for TensorBernoulli
bernoulli = TensorBernoulli(logits=torch.randn(2, 3, 4))

chain(bernoulli)  # Works perfectly fine!
