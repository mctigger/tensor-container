import torch
from torch.distributions import (
    Bernoulli,
    Categorical,
    Distribution,
    Normal,
    kl_divergence,
)


def partially_detached_kl_divergence(p: Distribution, q: Distribution):
    """
    Compute KL divergence between p and a detached version of q.

    With standard torch.distributions, we need type-specific handling
    because different distributions have different parameter names and structures.
    """
    # Create detached version of q based on its type
    if isinstance(q, Normal):
        detached_q = Normal(loc=q.loc.detach(), scale=q.scale.detach())
    elif isinstance(q, Categorical):
        detached_q = Categorical(logits=q.logits.detach())
    elif isinstance(q, Bernoulli):
        detached_q = Bernoulli(probs=q.probs.detach())
    else:
        raise RuntimeError(
            f"partially_detached_kl_divergence not implemented for distribution {type(q)}"
        )

    return kl_divergence(p, detached_q)


# Create different types of distributions with gradients
normal = Normal(
    loc=torch.tensor([0.0, 1.0], requires_grad=True),
    scale=torch.tensor([1.0, 0.5], requires_grad=True),
)
categorical = Categorical(
    logits=torch.tensor([[0.1, 0.9], [0.8, 0.2]], requires_grad=True)
)
bernoulli = Bernoulli(probs=torch.tensor([0.2, 0.8], requires_grad=True))

# Each distribution type requires the same function but with type-specific logic
kl_normal = partially_detached_kl_divergence(normal, normal)
kl_categorical = partially_detached_kl_divergence(categorical, categorical)
kl_bernoulli = partially_detached_kl_divergence(bernoulli, bernoulli)
