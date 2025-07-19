# TensorDistribution
TensorDistribution adds tensor-like properties to torch.distributions and enable application of torch operations like `torch.stack()` or `torch.cat()` for distributions.

## Key Benefits
- **🔄 Drop-in torch.distributions replacement**: tensorcontainer.tensor_distribution is API-compatible with torch.distributions
- **🏗️ TensorContainer integration**: Seamlessly works with TensorDict and TensorDataClass structures
- **⚡ Unified tensor operations**: Indexing, slicing, reshaping, and device transfer work on entire distributions
- **📦 Efficient batching**: Batch operations across multiple samples with consistent shape handling
- **🚀 torch.compile compatibility**: PyTree support enables torch.compile compatability

To see the benefits in action, here are two examples:

1. **Seamless Tensor Operations**: Apply `torch` operations like `view`, `permute`, and `detach` directly to `TensorDistribution` instances, simplifying batch and shape management.


```python
import torch

from tensorcontainer.tensor_distribution import TensorNormal


loc = torch.randn(2 * 3, 4)
scale = torch.abs(torch.randn(2 * 3, 4))
normal = TensorIndependent(TensorNormal(loc=loc, scale=scale))

normal = normal.view(2, 3).permute(1, 0, 2).detach()
```

2. **Distribution-Agnostic Operations**: Implement generic functions that work with any `TensorDistribution` type, such as computing KL divergence, without needing type-specific parameter handling. This contrasts with `torch.distributions`, which often requires explicit checks for each distribution type.

```python
import torch
from torch import nn
from torch.distributions import kl_divergence


class LossModule(nn.Module):
    def __init__(self, weight):
        self._weight = weight

    def forward(self, p: TensorDistribution, q: TensorDistribution):
        kl_p = kl_divergence(p, q.detach())
        kl_q = kl_divergence(p.detach(), q)

        return self._weight * kl_p + (1 - self._weight) * kl_q
```

---

`TensorDistribution` extends PyTorch's `torch.distributions` with tensor-like operations and structured data support. Instead of manually managing distribution parameters across different devices, batch dimensions, and nested structures, TensorDistribution provides a unified interface that works just like regular tensors.

---

## 1. What `TensorDistribution` Adds to `torch.distributions`

This section details how `TensorDistribution` enhances `torch.distributions` for structured, tensor-based workflows.

### Core Enhancement: `tensorcontainer` Integration

The main advantage of `TensorDistribution` is that it fully integrates `torch.distributions.Distribution` instances into the `tensorcontainer` ecosystem. By wrapping a distribution, you grant it tensor-like properties, including:

-   **Batching:** Treat a collection of distributions as a single, batch-aware entity.
-   **Indexing and Slicing:** Use standard `__getitem__` syntax to select subsets of your batched distributions.
-   **Shape Manipulation:** Apply operations like `.view()`, `.reshape()`, `.squeeze()`, and `.expand()` directly to the distribution batch.
-   **Device Portability:** Move distributions between devices (`.to()`, `.cuda()`, `.cpu()`) just like a tensor.

This is especially useful in areas like VAEs or RL, where models frequently produce distributions for structured data. `TensorDistribution` ensures these outputs can be handled with the same tooling as regular tensor data.

### When to Wrap with `TensorDistribution`

-   **Choose `TensorDistribution`** when your model's output is a *distribution* over a batch of data, and you need to apply tensor-like operations to that batch of distributions.
-   **Use `TensorDataClass` or `TensorDict`** when you are working with *concrete, realized data* that has already been sampled or observed.

---

## 2. Advantages of TensorDistribution over torch.distributions

This section demonstrates key advantages of using `TensorDistribution` compared to standard `torch.distributions`.

### Unified API Across Distribution Types

`TensorDistribution` provides a consistent interface for different distribution types, simplifying operations that would otherwise require type-specific handling with `torch.distributions`. This is particularly evident when performing operations like detaching parameters or applying transformations, where `TensorDistribution` abstracts away the underlying parameter structure.

Consider the task of computing the KL divergence between a distribution and its detached version. With `torch.distributions`, this requires explicit checks for each distribution type to access and detach its specific parameters:

```python
import torch
from torch.distributions import (
    Bernoulli,
    Categorical,
    Distribution,
    Normal,
    kl_divergence,
)


def partially_detached_kl_divergence_torch(p: Distribution, q: Distribution):
    """
    Compute KL divergence between p and a detached version of q using torch.distributions.
    Requires type-specific handling due to varying parameter names and structures.
    """
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

# Example usage with torch.distributions
normal_torch = Normal(
    loc=torch.tensor([0.0, 1.0], requires_grad=True),
    scale=torch.tensor([1.0, 0.5], requires_grad=True),
)
categorical_torch = Categorical(
    logits=torch.tensor([[0.1, 0.9], [0.8, 0.2]], requires_grad=True)
)
bernoulli_torch = Bernoulli(probs=torch.tensor([0.2, 0.8], requires_grad=True))

kl_normal_torch = partially_detached_kl_divergence_torch(normal_torch, normal_torch)
kl_categorical_torch = partially_detached_kl_divergence_torch(categorical_torch, categorical_torch)
kl_bernoulli_torch = partially_detached_kl_divergence_torch(bernoulli_torch, bernoulli_torch)
```

In contrast, `TensorDistribution` provides a unified `detach()` method that works seamlessly across all distribution types, eliminating the need for conditional logic:

```python
import torch
from torch.distributions import kl_divergence

from tensorcontainer.tensor_distribution import (
    TensorBernoulli,
    TensorCategorical,
    TensorDistribution,
    TensorNormal,
)


def partially_detached_kl_divergence_tensor(p: TensorDistribution, q: TensorDistribution):
    """
    Compute KL divergence between p and a detached version of q using TensorDistribution.
    Simply calls .detach() on the distribution, regardless of its specific type.
    """
    return kl_divergence(p, q.detach())

# Example usage with TensorDistribution
normal_tensor = TensorNormal(
    loc=torch.tensor([0.0, 1.0], requires_grad=True),
    scale=torch.tensor([1.0, 0.5], requires_grad=True),
)
categorical_tensor = TensorCategorical(
    logits=torch.tensor([[0.1, 0.9], [0.8, 0.2]], requires_grad=True)
)
bernoulli_tensor = TensorBernoulli(probs=torch.tensor([0.2, 0.8], requires_grad=True))

kl_normal_tensor = partially_detached_kl_divergence_tensor(normal_tensor, normal_tensor)
kl_categorical_tensor = partially_detached_kl_divergence_tensor(categorical_tensor, categorical_tensor)
kl_bernoulli_tensor = partially_detached_kl_divergence_tensor(bernoulli_tensor, bernoulli_tensor)
```

This unified API significantly reduces boilerplate code and improves the maintainability and extensibility of your probabilistic models, as new distribution types can be integrated without modifying existing generic functions.

### Operation Chaining

TensorDistribution enables seamless chaining of tensor operations on distributions, allowing for cleaner and more readable code. This is particularly beneficial when working with complex tensor transformations that would otherwise require manual parameter extraction and re-instantiation for each operation.

The following TensorDistribution example demonstrates how multiple tensor operations can be chained together in a single, unified function that works across different distribution types:

```python
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
```

In contrast, the equivalent torch.distributions approach requires separate functions for each operation, with type-specific parameter handling. Each transformation must manually extract parameters, apply the transformation, and reconstruct the distribution:

```python
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
```

The torch.distributions approach requires distribution-specific functions because each distribution type has different parameter names (`loc`/`scale` for Normal vs `logits` for Bernoulli). This leads to verbose, error-prone code that breaks when applied to different distribution types.

### Nested Distributions

TensorDistribution simplifies working with nested distributions, such as those created using `Independent`. This is particularly useful when you need to apply tensor transformations to distributions that have both batch and event dimensions, as TensorDistribution automatically handles the complexity of preserving event dimensions while transforming batch dimensions.

The following TensorDistribution example shows how nested distributions can be manipulated with a simple, direct interface:

```python
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
```

In contrast, the torch.distributions approach requires a multi-step process to achieve the same result. You must manually extract the base distribution, transform its parameters while carefully preserving event dimensions, and then reconstruct the nested structure:

```python
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
```

The torch.distributions approach requires explicit knowledge of the nested structure and careful handling of batch versus event dimensions. Each step must be performed manually, making the code verbose and error-prone, especially when working with complex nested distribution hierarchies.

---

## 3. How-To Guides (Practical Recipes)

This section provides goal-oriented examples of `TensorDistribution`'s added capabilities.

### How to Sample from a TensorDistribution

The `.sample()` and `.rsample()` methods mirror their `torch.distributions` counterparts but are called on the `TensorDistribution` wrapper. They return a `torch.Tensor` of sampled values.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

# Create a batch of Normal distributions using TensorNormal
means = torch.tensor([0.0, 1.0])
std_devs = torch.tensor([1.0, 0.5])
tdist = TensorNormal(loc=means, scale=std_devs)

# Sample from the distribution
sample = tdist.sample()
# sample is a torch.Tensor
# Values will vary

# Resample (reparameterized sample)
rsample = tdist.rsample()
# rsample is a torch.Tensor
# Values will vary
```

### How to Compute Log Probability

The `.log_prob()` method works as expected, computing the log probability of a given sample.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

# Create a batch of Normal distributions
means = torch.tensor([0.0, 1.0])
std_devs = torch.tensor([1.0, 0.5])
tdist = TensorNormal(loc=means, scale=std_devs)

# Generate some samples (can be from tdist.sample() or external)
samples = torch.tensor([-0.1, 1.2])

# Compute log probability
log_probs = tdist.log_prob(samples)
# log_probs is a tensor with log probabilities
# Values will vary
```

### How to Apply Tensor-like Transformations

This is a key advantage of `TensorDistribution`. Operations like `clone()`, `to(device)`, `cpu()`, and `cuda()` are applied to all underlying parameters of the distribution simultaneously.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

mean = torch.tensor([0.0, 1.0])
stddev = torch.tensor([1.0, 0.5])
tdist = TensorNormal(loc=mean, scale=stddev)

# Clone the parameter tensors
cloned_tdist = tdist.clone()
# Move parameter to GPU
cuda_tdist = tdist.to("cuda")
```

### How to Index and Slice a TensorDistribution

Standard `__getitem__` indexing can be used to slice the batch dimension of your distribution. This operation is applied uniformly to the underlying parameters.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

means = torch.tensor([0.0, 1.0, 2.0])
std_devs = torch.tensor([1.0, 0.5, 0.2])
tdist = TensorNormal(loc=means, scale=std_devs)

# Original batch shape is torch.Size([3])

# Slicing the first element
sliced_tdist = tdist[0]
# sliced_tdist.shape is torch.Size([])

# Slicing a range
range_sliced_tdist = tdist[1:3]
# range_sliced_tdist.shape is torch.Size([2])
```

### How to Manipulate Batch Shapes

`TensorDistribution` provides tensor-like methods (`view()`, `reshape()`, `permute()`, `squeeze()`, `unsqueeze()`, `expand()`) to manipulate the batch shape of the underlying distribution's parameters.

```python
import torch
from torch.distributions import Normal
from tensorcontainer.tensor_distribution import TensorNormal

means = torch.randn(1, 4, 5)
std_devs = torch.ones(1, 4, 5)
tdist = TensorNormal(loc=means, scale=std_devs)

# Original batch_size: torch.Size([1, 4, 5])

# Reshape the batch dimensions
reshaped_tdist = tdist.reshape(4, 5)
# Reshaped batch_size: torch.Size([4, 5])

# Squeeze a dimension
squeezed_tdist = tdist.squeeze(0)
# Squeezed batch_size: torch.Size([4, 5])
```
---

## 4. API Reference

This section provides a technical description of the `TensorDistribution` API.

### Initialization & Core Attributes

*   `__init__(self, dist: torch.distributions.Distribution)`: Constructor. Wraps a `torch.distributions.Distribution` instance.
*   `dist`: The underlying `torch.distributions.Distribution` instance.
*   `shape`: The `torch.Size` of the batch dimensions of the wrapped distribution.
*   `device`: The `torch.device` where the distribution's parameters reside.

### Sampling Methods

*   `sample(self, sample_shape=torch.Size())`: Generates samples. Returns a `torch.Tensor`.
*   `rsample(self, sample_shape=torch.Size())`: Generates reparameterized samples. Returns a `torch.Tensor`.

### Probability Methods

*   `log_prob(self, value)`: Returns the log probability of `value`.

### Tensor-like Operations

*   `clone()`: Returns a deep copy.
*   `to(device)`: Moves parameters to the specified device.
*   `cpu()`: Moves parameters to CPU memory.
*   `cuda()`: Moves parameters to CUDA memory.
*   `detach()`: Detaches parameters from the computation graph.

### Shape Manipulation

*   `view(*shape)`: Reshapes batch dimensions.
*   `reshape(*shape)`: Reshapes batch dimensions (non-contiguous memory safe).
*   `permute(*dims)`: Permutes batch dimensions.
*   `squeeze(*dims)`: Removes singleton dimensions.
*   `unsqueeze(*dims)`: Adds singleton dimensions.
*   `expand(*sizes)`: Expands batch dimensions.

---

## 5. Limitations

*   **Implementation-Dependent:** Behavior is determined by the underlying `torch.distributions.Distribution` instance.
*   **Parameter Immutability:** `TensorDistribution` methods should be preferred for transformations over direct manipulation of distribution parameters (e.g., `tdist.dist.loc = new_loc`), which can cause inconsistencies.