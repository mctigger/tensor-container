import torch

from tensorcontainer.tensor_distribution.independent import TensorIndependent
from tensorcontainer.tensor_distribution.normal import TensorNormal

# Create a TensorNormal with one event dimension
loc = torch.randn(2 * 3, 4)
scale = torch.abs(torch.randn(2 * 3, 4))

# Use TensorIndependent to create a TensorNormal with one event dimension
independent_normal = TensorIndependent(TensorNormal(loc=loc, scale=scale), 1)

# We do not need to care about Independent or even the type of distribution that
# Independent wraps, it just works. The last dimension is an event dimension
# so we must not pass it to .view()
independent_normal = independent_normal.view(2, 3)
