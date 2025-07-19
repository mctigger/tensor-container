import torch
from torch.distributions import kl_divergence

from tensorcontainer.tensor_distribution import (
    TensorBernoulli,
    TensorCategorical,
    TensorDistribution,
    TensorNormal,
)


def partially_detached_kl_divergence(p: TensorDistribution, q: TensorDistribution):
    """
    Compute KL divergence between p and a detached version of q.

    With TensorDistribution, we can simply call .detach() on any distribution
    without needing to know its specific type or parameter names.
    """
    return kl_divergence(p, q.detach())


# Create different types of TensorDistributions with gradients
normal = TensorNormal(
    loc=torch.tensor([0.0, 1.0], requires_grad=True),
    scale=torch.tensor([1.0, 0.5], requires_grad=True),
)
categorical = TensorCategorical(
    logits=torch.tensor([[0.1, 0.9], [0.8, 0.2]], requires_grad=True)
)
bernoulli = TensorBernoulli(probs=torch.tensor([0.2, 0.8], requires_grad=True))

# The same function works for all distribution types
# No type-specific handling required!
kl_normal = partially_detached_kl_divergence(normal, normal)
kl_categorical = partially_detached_kl_divergence(categorical, categorical)
kl_bernoulli = partially_detached_kl_divergence(bernoulli, bernoulli)
