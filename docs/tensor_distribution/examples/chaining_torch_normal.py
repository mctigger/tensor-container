import torch
from torch.distributions import Bernoulli, Normal


# Extract parameters, transform them, create new distribution
def view(normal):
    # Careful! Do not change the event dimension!
    viewed_loc = normal.loc.view(2, 3, 4)
    viewed_scale = normal.scale.view(2, 3, 4)
    return Normal(loc=viewed_loc, scale=viewed_scale)


# Extract parameters, permute them, create new distribution
def permute(normal):
    # Careful! Do not change the event dimension!
    permuted_loc = normal.loc.permute(1, 0, 2)
    permuted_scale = normal.scale.permute(1, 0, 2)
    return Normal(loc=permuted_loc, scale=permuted_scale)


# Extract parameters, detach them, create new distribution
def detach(normal):
    detached_loc = normal.loc.detach()
    detached_scale = normal.scale.detach()
    return Normal(loc=detached_loc, scale=detached_scale)


def chain(normal):
    normal = view(normal)
    normal = permute(normal)
    normal = detach(normal)

    return normal


# Create a for torch.distributions.Normal with one event dimension
# For the purposes of this tutorial we do not use Independent, although it
# would make sense here. See the section on Independent.
loc = torch.randn(2 * 3 * 4)
scale = torch.abs(torch.randn(2 * 3 * 4))
normal = Normal(loc=loc, scale=scale)

# Execute the chain for torch.distributions.Normal
chain(normal)

# Try to execute the chain for torch.distributions.Bernoulli
bernoulli = Bernoulli(logits=torch.randn(2, 3, 4))

chain(bernoulli)  # AttributeError: 'Bernoulli' object has no attribute 'loc'
