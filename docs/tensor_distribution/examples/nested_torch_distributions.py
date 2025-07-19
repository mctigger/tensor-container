import torch
from torch.distributions import Independent, Normal

loc = torch.randn(2 * 3, 4)
scale = torch.abs(torch.randn(2 * 3, 4))

# Use Independent to create a Normal with one event dimension
normal = Independent(Normal(loc=loc, scale=scale), 1)

# 1. Extract the base distribution
base_dist = normal.base_dist

# 2. Extract the parameters
loc = base_dist.loc
scale = base_dist.scale

# 3. Reshape the parameters
# Note that we can't touch the event dimension, so we only reshape the batch dimensions
new_loc = loc.view(2, 3, 4)
new_scale = scale.view(2, 3, 4)

# 4. Create a new distribution
new_normal = Independent(Normal(loc=new_loc, scale=new_scale), 1)
