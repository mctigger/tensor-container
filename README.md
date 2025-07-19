# Tensor Container

*Tensor containers for PyTorch with PyTree compatibility and torch.compile optimization*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **⚠️ Academic Research Project**: This project exists solely for academic purposes to explore and learn PyTorch internals. For production use, please use the official, well-maintained [**torch/tensordict**](https://github.com/pytorch/tensordict) library.

Tensor Container provides efficient, type-safe tensor container implementations for PyTorch workflows. It includes PyTree integration and torch.compile optimization for batched tensor operations.

The library includes tensor containers, probabilistic distributions, and batch/event dimension semantics for machine learning workflows.

## What is TensorContainer?

TensorContainer transforms how you work with structured tensor data in PyTorch by providing **tensor-like operations for entire data structures**. Instead of manually managing individual tensors across devices, batch dimensions, and nested hierarchies, TensorContainer lets you treat complex data as unified entities that behave just like regular tensors.

### 🚀 **Unified Operations Across Data Types**

Apply tensor operations like `view()`, `permute()`, `detach()`, and device transfers to entire data structures—no matter how complex:

```python
# Single operation transforms entire distribution
distribution = distribution.view(2, 3, 4).permute(1, 0, 2).detach()

# Works seamlessly across TensorDict, TensorDataClass, and TensorDistribution
data = data.to('cuda').reshape(batch_size, -1).clone()
```

### 🔄 **Drop-in Compatibility with PyTorch**

TensorContainer integrates seamlessly with existing PyTorch workflows:
- **torch.distributions compatibility**: TensorDistribution is API-compatible with `torch.distributions` while adding tensor-like operations
- **PyTree support**: All containers work with `torch.utils._pytree` operations and `torch.compile`
- **Zero learning curve**: If you know PyTorch tensors, you already know TensorContainer

### ⚡ **Eliminates Boilerplate Code**

Compare the complexity difference:

**With torch.distributions** (manual parameter handling):
```python
# Requires type-specific parameter extraction and reconstruction
if isinstance(dist, Normal):
    detached = Normal(loc=dist.loc.detach(), scale=dist.scale.detach())
elif isinstance(dist, Categorical):
    detached = Categorical(logits=dist.logits.detach())
# ... more type checks needed
```

**With TensorDistribution** (unified interface):
```python
# Works for any distribution type
detached = dist.detach()
```

### 🏗️ **Structured Data Made Simple**

Handle complex, nested tensor structures with the same ease as single tensors:
- **Batch semantics**: Consistent shape handling across all nested tensors
- **Device management**: Move entire structures between CPU/GPU with single operations
- **Shape validation**: Automatic verification of tensor compatibility
- **Type safety**: Full IDE support with static typing and autocomplete

TensorContainer doesn't just store your data—it makes working with structured tensors as intuitive as working with individual tensors, while maintaining full compatibility with the PyTorch ecosystem you already know.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [API Overview](#api-overview)
- [torch.compile Compatibility](#torchcompile-compatibility)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)
- [Authors](#authors)
- [Contact and Support](#contact-and-support)

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/mctigger/tensor-container.git
cd tensor-container

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev]
```

### Requirements

- Python 3.9+
- PyTorch 2.0+

## Quick Start

### TensorDict: Dictionary-Style Containers

```python
import torch
from tensorcontainer import TensorDict

# Create a TensorDict with batch semantics
data = TensorDict({
    'observations': torch.randn(32, 128),
    'actions': torch.randn(32, 4),
    'rewards': torch.randn(32, 1)
}, shape=(32,), device='cpu')

# Dictionary-like access
obs = data['observations']
data['new_field'] = torch.zeros(32, 10)

# Batch operations work seamlessly
stacked_data = torch.stack([data, data])  # Shape: (2, 32)
```

### TensorDataClass: Type-Safe Containers

```python
import torch
from tensorcontainer import TensorDataClass

class RLData(TensorDataClass):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor

# Create with full type safety and IDE support
data = RLData(
    observations=torch.randn(32, 128),
    actions=torch.randn(32, 4),
    rewards=torch.randn(32, 1),
    shape=(32,),
    device='cpu'
)

# Type-safe field access with autocomplete
obs = data.observations
data.actions = torch.randn(32, 8)  # Type-checked assignment
```

### TensorDistribution: Probabilistic Containers

```python
import torch
from tensorcontainer import TensorDistribution

# Built-in distribution types
from tensorcontainer.tensor_distribution import (
    TensorNormal, TensorBernoulli, TensorCategorical,
    TensorTruncatedNormal, TensorTanhNormal
)

# Create probabilistic tensor containers
normal_dist = TensorNormal(
    loc=torch.zeros(32, 4),
    scale=torch.ones(32, 4),
    shape=(32,),
    device='cpu'
)

# Sample and compute probabilities
samples = normal_dist.sample()  # Shape: (32, 4)
log_probs = normal_dist.log_prob(samples)
entropy = normal_dist.entropy()

# Categorical distributions for discrete actions
categorical = TensorCategorical(
    logits=torch.randn(32, 6),  # 6 possible actions
    shape=(32,),
    device='cpu'
)
```

### PyTree Operations

```python
# All containers work seamlessly with PyTree operations
import torch.utils._pytree as pytree

# Transform all tensors in the container
doubled_data = pytree.tree_map(lambda x: x * 2, data)

# Combine multiple containers
combined = pytree.tree_map(lambda x, y: x + y, data1, data2)
```

## Features

- **torch.compile Optimized**: Compatible with PyTorch's JIT compiler
- **PyTree Support**: Integration with `torch.utils._pytree` for tree operations
- **Zero-Copy Operations**: Efficient tensor sharing and manipulation
- **Type Safety**: Static typing support with IDE autocomplete and type checking
- **Batch Semantics**: Consistent batch/event dimension handling
- **Shape Validation**: Automatic validation of tensor shapes and device consistency
- **Multiple Container Types**: Different container types for different use cases
- **Probabilistic Support**: Distribution containers for probabilistic modeling
- **Comprehensive Testing**: Extensive test suite with compile compatibility verification
- **Memory Efficient**: Optimized memory usage with slots-based dataclasses

## API Overview

### Core Components

- **`TensorContainer`**: Base class providing core tensor manipulation operations with batch/event dimension semantics
- **`TensorDict`**: Dictionary-like container for dynamic tensor collections with nested structure support
- **`TensorDataClass`**: DataClass-based container for static, typed tensor structures
- **`TensorDistribution`**: Distribution wrapper for probabilistic tensor operations

### Key Concepts

- **Batch Dimensions**: Leading dimensions defined by the `shape` parameter, consistent across all tensors
- **Event Dimensions**: Trailing dimensions beyond batch shape, can vary per tensor
- **PyTree Integration**: All containers are registered PyTree nodes for seamless tree operations
- **Device Consistency**: Automatic validation ensures all tensors reside on compatible devices
- **Unsafe Construction**: Context manager for performance-critical scenarios with validation bypass

## torch.compile Compatibility

Tensor Container is designed for `torch.compile` compatibility:

```python
@torch.compile
def process_batch(data: TensorDict) -> TensorDict:
    # PyTree operations compile efficiently
    return TensorContainer._tree_map(lambda x: torch.relu(x), data)

@torch.compile
def sample_and_score(dist: TensorNormal, actions: torch.Tensor) -> torch.Tensor:
    # Distribution operations are compile-safe
    return dist.log_prob(actions)

# All operations compile efficiently with minimal graph breaks
compiled_result = process_batch(tensor_dict)
log_probs = sample_and_score(normal_dist, action_tensor)
```

The testing framework includes compile compatibility verification to ensure operations work efficiently under JIT compilation, including:
- Graph break detection and minimization
- Recompilation tracking
- Memory leak prevention
- Performance benchmarking

## Contributing

Contributions are welcome! Tensor Container is a learning project for exploring PyTorch internals and tensor container implementations.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/mctigger/tensor-container.git
cd tensor-container
pip install -e .[dev]
```

### Running Tests

```bash
# Run all tests with coverage
pytest --strict-markers --cov=src

# Run specific test modules
pytest tests/tensor_dict/test_compile.py
pytest tests/tensor_dataclass/
pytest tests/tensor_distribution/

# Run compile-specific tests
pytest tests/tensor_dict/test_graph_breaks.py
pytest tests/tensor_dict/test_recompilations.py
```

### Development Guidelines

- All new features must maintain `torch.compile` compatibility
- Comprehensive tests required, including compile compatibility verification
- Follow existing code patterns and typing conventions
- Distribution implementations must support KL divergence registration
- Memory efficiency considerations for large-scale tensor operations
- Unsafe construction patterns for performance-critical paths

### Contribution Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with appropriate tests
4. Ensure all tests pass and maintain coverage
5. Submit a pull request with a clear description

## Documentation

The project includes documentation:

- **[`docs/compatibility.md`](docs/compatibility.md)**: Python version compatibility guide and best practices
- **[`docs/testing.md`](docs/testing.md)**: Testing philosophy, standards, and guidelines
- **Source Code Documentation**: Extensive docstrings and type annotations throughout the codebase
- **Test Coverage**: 643+ tests covering all major functionality with 86% code coverage

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Tim Joseph** - [mctigger](https://github.com/mctigger)

## Contact and Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/mctigger/tensor-container/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/mctigger/tensor-container/discussions)
- **Email**: For direct inquiries, contact [tim@mctigger.com](mailto:tim@mctigger.com)

---

*Tensor Container is an academic research project for learning PyTorch internals and tensor container patterns. For production applications, we strongly recommend using the official [torch/tensordict](https://github.com/pytorch/tensordict) library, which is actively maintained by the PyTorch team.*