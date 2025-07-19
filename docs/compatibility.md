# Python Compatibility Guide

This document outlines compatibility considerations and solutions for supporting different Python versions in the tensorcontainer project.

## Supported Python Versions

**tensorcontainer requires Python 3.9 or higher.**

## Compatibility Changes

### Union Types and isinstance() Checks

Union types can now be handled using `typing.get_args()`:

```python
from typing import Union, get_args
import torch
from tensorcontainer import TensorContainer

TDCompatible = Union[torch.Tensor, TensorContainer]

# Use get_args to extract types from Union:
if isinstance(val, get_args(TDCompatible)):
    pass
```

### Type Annotations

**Note**: The `|` operator for union types was introduced in Python 3.10:

```python
# Python 3.10+ only:
def func(x: int | str) -> None:
    pass
```

For broader compatibility with Python 3.9+, use `Union` from typing:

```python
from typing import Union

# Compatible with Python 3.9+:
def func(x: Union[int, str]) -> None:
    pass
```

## General Compatibility Tips

### Import Compatibility

Use `from __future__ import annotations` at the top of files to enable forward references and improve compatibility:

```python
from __future__ import annotations

# This allows using string annotations that are evaluated later
def func(x: 'SomeClass') -> 'SomeClass':
    pass
```

### Version-Specific Features

When using features that are only available in newer Python versions, use version checks:

```python
import sys

if sys.version_info >= (3, 10):
    # Use Python 3.10+ features
    pass
else:
    # Fallback for Python 3.9
    pass
```

Version-specific features should be avoided whenever possible. Instead, use one of the above solutions.