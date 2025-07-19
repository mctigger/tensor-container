# TensorDataClass User Guide

`TensorDataClass` is a dataclass-based container for PyTorch tensors, designed to provide strong typing, IDE support, and structured data management for batched tensor operations. It automatically converts annotated classes into dataclasses with fixed schemas.

---

## 1. Explanation (The "Big Picture")

This section explains the core ideas behind `TensorDataClass` to help you understand its purpose and design.

### Core Concepts

`TensorDataClass` is a dataclass-based container specifically designed for `torch.Tensor` objects. It provides a fixed schema, allowing for attribute-style access to its contained tensors. A key feature is its automatic handling of batch dimensions, ensuring consistency across all tensors within an instance. It seamlessly integrates with `torch.compile` and PyTorch's PyTree system, making it a powerful tool for managing structured tensor data in deep learning workflows. It solves the problem of managing heterogeneous tensor data in a type-safe and readable manner, particularly in domains such as reinforcement learning, batch processing, or applications involving structured data where each field is a tensor.

### When to use TensorDataClass vs. TensorDict?

| Feature | TensorDataClass | TensorDict |
| :---------------- | :-------------------------------------------- | :-------------------------------------------- |
| **Access Pattern** | `obj.field` (attribute access) | `obj["key"]` (dictionary-style access) |
| **Type Safety** | Static typing, IDE autocomplete, compile-time checks | Runtime checks, less IDE support for keys |
| **IDE Support** | Full autocomplete, type hints | Limited (keys are strings, not attributes) |
| **Memory Usage** | Lower (`slots=True` by default) | Higher (uses a dictionary for storage) |
| **Field Definition** | Compile-time (defined in class) | Runtime (can add/remove keys dynamically) |
| **Inheritance** | Natural OOP inheritance patterns | Composition (can contain other `TensorDict`s) |
| **Dynamic Fields** | Not supported | Full support |
| **Use Case** | Fixed, well-defined data schemas | Flexible, dynamic key-value tensor storage |

**When to choose `TensorDataClass`:**

*   When your data has a fixed, known structure (schema) that won't change at runtime.
*   When type safety, IDE autocomplete, and static analysis are high priorities for code maintainability and readability.
*   When you want to leverage Python's dataclass features and OOP inheritance for your tensor containers.
*   When memory efficiency and faster attribute access are critical.

**When to choose `TensorDict`:**

*   When you need a flexible container where the keys (and thus the contained tensors) can be added, removed, or changed dynamically at runtime.
*   When the structure of your data is not strictly fixed or can vary.
*   When you prefer a dictionary-like interface for managing your tensors.
*   When you need to store arbitrary key-value pairs where values are tensors or other `TensorDict`s.

---

## 2. Tutorial: Your First TensorDataClass

This tutorial will guide you through creating and using a basic `TensorDataClass`.

### Installation

`TensorDataClass` is part of the `tensorcontainer` library. Please refer to the main installation guide for instructions on how to install the package.

### A Hands-On Example

To define your first `TensorDataClass`, you simply create a Python class and decorate it with `@dataclass`, inheriting from `TensorDataClass`. You then annotate your fields with their respective types, typically `torch.Tensor`. Non-tensor fields are also supported.

```python
from dataclasses import dataclass
import torch
from tensorcontainer import TensorDataClass

@dataclass
class MyData(TensorDataClass):
    a: torch.Tensor
    b: torch.Tensor
    name: str = "default"

# Instantiate your TensorDataClass
data_instance = MyData(a=torch.tensor([1, 2]), b=torch.tensor([3, 4]), name="example")
# >>> MyData(a=tensor([1, 2]), b=tensor([3, 4]), name='example', batch_size=torch.Size([2]))

# Access fields using attribute-style access
data_instance.a
# >>> tensor([1, 2])
data_instance.b
# >>> tensor([3, 4])
data_instance.name
# >>> 'example'
```

This example demonstrates how `TensorDataClass` leverages Python's dataclass features to provide a clear and structured way to define your tensor containers.

---

## 3. How-To Guides (Practical Recipes)

This section provides a collection of goal-oriented guides to solve specific problems.

### How to Apply Tensor-like Transformations

`TensorDataClass` instances support common tensor operations like `clone()`, `to(device)`, `cpu()`, and `cuda()`. These operations are applied uniformly to all contained tensors, ensuring consistency.

```python
import torch
from dataclasses import dataclass
from tensorcontainer import TensorDataClass

@dataclass
class TransformableData(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor

transform_data = TransformableData(x=torch.randn(2, 3), y=torch.randn(2, 3))

# Clone the instance
# Clone the instance
cloned_data = transform_data.clone()
print(f"Cloned successfully: {cloned_data.x.shape == transform_data.x.shape and id(cloned_data.x) is not id(transform_data.x)}")
# >>> Cloned successfully: True

# Transfer to a different device (e.g., CUDA if available, otherwise CPU)
if torch.cuda.is_available():
    cuda_data = transform_data.to("cuda")
    print(f"Device: {cuda_data.device}, Child device: {cuda_data.x.device}")
    # >>> Device: cuda:0, Child device: cuda:0
else:
    print("CUDA not available. Skipping .to('cuda') example.")
    print("Note: .to('cuda') requires a CUDA-enabled device.")

cpu_data = transform_data.to("cpu")
print(f"Device: {cpu_data.device}, Child device: {cpu_data.x.device}")
# >>> Device: cpu, Child device: cpu
```

### How to Index and Slice Data

You can index and slice `TensorDataClass` instances using standard Python indexing (`__getitem__`). This operation applies the indexing uniformly across all contained tensors.

```python
import torch
from dataclasses import dataclass
from tensorcontainer import TensorDataClass

@dataclass
class IndexableData(TensorDataClass):
    x: torch.Tensor
    y: torch.Tensor

index_data = IndexableData(x=torch.randn(5, 3), y=torch.randn(5, 3))

# Slicing the first element
sliced_data = index_data[0]
print(f"Sliced data shape: {sliced_data.shape}, tensor shape: {sliced_data.x.shape}")
# >>> Sliced data shape: torch.Size([]), tensor shape: torch.Size([3])

# Slicing a range
range_sliced_data = index_data[1:3]
print(f"Range sliced data shape: {range_sliced_data.shape}, tensor shape: {range_sliced_data.x.shape}")
# >>> Range sliced data shape: torch.Size([2]), tensor shape: torch.Size([2, 3])
```

### How to Manipulate Batch Shapes

`TensorDataClass` provides methods like `view()`, `reshape()`, `permute()`, `squeeze()`, `unsqueeze()`, and `expand()` to manipulate the batch shapes of all contained tensors simultaneously.

```python
import torch
from dataclasses import dataclass
from tensorcontainer import TensorDataClass

@dataclass
class ShapeData(TensorDataClass):
    data: torch.Tensor
    other: torch.Tensor

shape_instance = ShapeData(data=torch.randn(1, 4, 5), other=torch.randn(1, 4, 5))

# Reshape the batch dimensions
reshaped_data = shape_instance.reshape(4, 1) # Note: adjusted shape to be valid
print(f"Reshaped batch_size: {reshaped_data.shape}")
# >>> Reshaped batch_size: torch.Size([4, 1])

# Squeeze a dimension
squeezed_data = shape_instance.squeeze(0)
print(f"Squeezed batch_size: {squeezed_data.shape}")
# >>> Squeezed batch_size: torch.Size([4])
```

### How to Combine Multiple Instances

`TensorDataClass` instances can be combined using `torch.stack()` and `torch.cat()`, which operate on the contained tensors while preserving the `TensorDataClass` structure.

```python
import torch
from dataclasses import dataclass
from tensorcontainer import TensorDataClass

@dataclass
class StackableData(TensorDataClass):
    val1: torch.Tensor
    val2: torch.Tensor

stack_instance1 = StackableData(val1=torch.tensor([1, 2]), val2=torch.tensor([10, 20]))
stack_instance2 = StackableData(val1=torch.tensor([3, 4]), val2=torch.tensor([30, 40]))

# Stack multiple instances
stacked_data = torch.stack([stack_instance1, stack_instance2], dim=0)
# >>> Stacked data batch_size: torch.Size([2, 2])
```

### How to Handle Non-Tensor Metadata

`TensorDataClass` allows for non-tensor fields (metadata). These fields are preserved during tensor operations but are generally ignored by tensor-specific transformations.

```python
import torch
from dataclasses import dataclass
from tensorcontainer import TensorDataClass

@dataclass
class MetadataData(TensorDataClass):
    tensor_field: torch.Tensor
    metadata: str

meta_instance = MetadataData(tensor_field=torch.randn(2, 2), metadata="important_info")
# >>> Original instance:
# >>> MetadataData(tensor_field=tensor([[ 0.0749,  0.0500],
# >>>                                    [-0.0200,  0.0749]]), metadata='important_info')

# Perform a tensor operation (e.g., .sum()). Metadata is preserved.
summed_instance = meta_instance.sum()
# >>> Summed instance:
# >>> MetadataData(tensor_field=tensor(0.0849), metadata='important_info')
# >>> Metadata preserved: important_info
```

### How to Create Nested and Inherited Structures

`TensorDataClass` fully supports both nesting (a `TensorDataClass` containing another `TensorDataClass`) and inheritance, allowing for complex and organized data structures. Operations are applied recursively to nested instances.

```python
import torch
from dataclasses import dataclass
from tensorcontainer import TensorDataClass

# Nested Structure Example
@dataclass
class InnerData(TensorDataClass):
    inner_tensor: torch.Tensor

@dataclass
class OuterData(TensorDataClass):
    outer_tensor: torch.Tensor
    nested: InnerData

nested_instance = OuterData(
    outer_tensor=torch.tensor([100, 200]),
    nested=InnerData(inner_tensor=torch.tensor([1, 2, 3]))
)
# >>> Nested instance:
# >>> OuterData(
# >>>     outer_tensor=tensor([100, 200]),
# >>>     nested=InnerData(inner_tensor=tensor([1, 2, 3])))
# >>> Accessing inner tensor: tensor([1, 2, 3])

# Inheritance Example
@dataclass
class BaseData(TensorDataClass):
    base_tensor: torch.Tensor
    base_info: str

@dataclass
class ExtendedData(BaseData):
    extended_tensor: torch.Tensor
    extended_info: int

extended_instance = ExtendedData(
    base_tensor=torch.tensor([5, 6]),
    base_info="from_base",
    extended_tensor=torch.tensor([7, 8]),
    extended_info=123
)
# >>> Extended instance:
# >>> ExtendedData(
# >>>     base_tensor=tensor([5, 6]),
# >>>     base_info='from_base',
# >>>     extended_tensor=tensor([7, 8]),
# >>>     extended_info=123)
# >>> Accessing base info: from_base
# >>> Accessing extended tensor: tensor([7, 8])
```

### How to Ensure Data Integrity

`TensorDataClass` enforces consistent batch shapes and devices across all its contained tensors. If you attempt to create an instance with inconsistent batch dimensions, a `ValueError` will be raised.

```python
import torch
from dataclasses import dataclass
from tensorcontainer import TensorDataClass

@dataclass
class ValidatedData(TensorDataClass):
    field1: torch.Tensor
    field2: torch.Tensor

try:
    # This will raise a ValueError because batch shapes are inconsistent
    invalid_instance = ValidatedData(field1=torch.randn(2, 3), field2=torch.randn(3, 3))
    # >>> Unexpectedly created instance: ValidatedData(field1=tensor([[ 0.0749,  0.0500, -0.0200],
    # >>>                                    [-0.0749,  0.0500, -0.0200]]), field2=tensor([[ 0.0749,  0.0500, -0.0200],
    # >>>                                                                               [-0.0749,  0.0500, -0.0200]]))
    except ValueError as e:
        # >>> Caught expected error: Inconsistent batch sizes: expected torch.Size([2, 3]) but got torch.Size([3, 3])
```

---

## 4. API Reference

This section provides a technical, exhaustive description of the `TensorDataClass` API.

### Initialization & Core Attributes

*   `__init__(self, shape, device, **kwargs)`: Constructor for `TensorDataClass`.
    *   `shape`: The batch shape of the `TensorDataClass` instance.
    *   `device`: The device where the tensors are located.
    *   `**kwargs`: Keyword arguments corresponding to the fields defined in the dataclass.
*   `shape`: Property returning the `torch.Size` of the batch dimensions.
*   `device`: Property returning the `torch.device` where the tensors reside.

### Special Methods (Magic Methods)

*   `__getitem__`: Supports indexing and slicing of the `TensorDataClass` instance, applying the operation uniformly to all contained tensors.
*   `__setitem__`: Supports setting values using indexing, allowing for in-place modification of contained tensors.

### Tensor-like Operations

*   `clone()`: Returns a deep copy of the `TensorDataClass` instance and its contained tensors.
*   `to(device)`: Moves all contained tensors to the specified device.
*   `cpu()`: Moves all contained tensors to CPU memory.
*   `cuda()`: Moves all contained tensors to CUDA memory (if available).
*   `detach()`: Returns a new `TensorDataClass` instance with all contained tensors detached from the current computation graph.

### Shape Manipulation

*   `view(*shape)`: Returns a new `TensorDataClass` instance with the batch dimensions reshaped according to the provided `shape`.
*   `reshape(*shape)`: Similar to `view`, but can handle non-contiguous memory.
*   `permute(*dims)`: Permutes the dimensions of all contained tensors.
*   `squeeze(*dims)`: Removes singleton dimensions from the batch shape.
*   `unsqueeze(*dims)`: Adds singleton dimensions to the batch shape.
*   `expand(*sizes)`: Expands the batch dimensions of all contained tensors.

### Class Methods for Combining Instances

*   `stack(list_of_instances, dim=0)`: Stacks a list of `TensorDataClass` instances along a new dimension.
*   `cat(list_of_instances, dim=0)`: Concatenates a list of `TensorDataClass` instances along an existing dimension.

---

## 5. Limitations

*   **Fixed Schema:** Unlike `TensorDict`, `TensorDataClass` does not support dynamic addition or removal of fields at runtime. Its structure is defined at compile-time through class annotations.
*   **No `eq=True`:** `TensorDataClass` instances cannot be compared for equality using `==` (i.e., `eq=False` is enforced). This is due to the complexities and potential performance issues of comparing floating-point tensors for exact equality.
*   **Not for Arbitrary Key-Value Storage:** It's not intended as a general-purpose dictionary for tensors where keys are arbitrary and can change.