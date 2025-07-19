# TensorAnnotated Usage Guide

`TensorAnnotated` is a powerful base class designed to facilitate the creation of custom data structures that seamlessly integrate with PyTorch's PyTree mechanism. By subclassing `TensorAnnotated` and using type annotations for your attributes, you can define complex objects that behave like native PyTorch tensors in operations such as `copy()`, `to()`, `cuda()`, and more.

## What `TensorAnnotated` Offers

The core purpose of `TensorAnnotated` is to enable automatic PyTree flattening and unflattening for custom Python classes. This means:

*   **Automatic Tensor Handling:** Any attribute type-annotated as a `torch.Tensor` or another `TensorContainer` (like `TensorDict` or another `TensorAnnotated` instance) will be automatically included in PyTree operations. This allows for easy movement of data between devices, cloning, and other tensor-centric manipulations.
*   **Structured Data with PyTorch Integration:** You can define rich, domain-specific data structures (e.g., a `RobotState` class with `position: Tensor`, `velocity: Tensor`, `joint_angles: Tensor`) that still benefit from PyTorch's ecosystem.
*   **Metadata Preservation:** Attributes that are type-annotated but are *not* tensors (e.g., integers, strings, lists) are treated as metadata and are preserved during PyTree operations, ensuring your object's non-tensor state is maintained.

## How to Use `TensorAnnotated`

To use `TensorAnnotated`, you need to subclass it and define your tensor attributes using type annotations.

### 1. Subclass `TensorAnnotated`

Begin by inheriting from `TensorAnnotated`.

```python
from torch import Tensor
from tensorcontainer.tensor_annotated import TensorAnnotated

class MyCustomData(TensorAnnotated):
    # ... define attributes and __init__
    pass
```

### 2. Define Annotated Attributes

Declare your attributes with type hints. For attributes you want to be part of PyTree operations, use `torch.Tensor` or `TensorContainer` types. For metadata, use any other Python type.

```python
from torch import Tensor
from tensorcontainer.tensor_annotated import TensorAnnotated

class MyCustomData(TensorAnnotated):
    my_tensor: Tensor
    my_other_tensor: Tensor
    my_metadata: str
    my_number: int

    def __init__(self, my_tensor: Tensor, my_other_tensor: Tensor, my_metadata: str, my_number: int):
        self.my_tensor = my_tensor
        self.my_other_tensor = my_other_tensor
        self.my_metadata = my_metadata
        self.my_number = my_number
        # IMPORTANT: Call super().__init__
        super().__init__(shape=my_tensor.shape, device=my_tensor.device)

# Example Instantiation
import torch
data_instance = MyCustomData(
    my_tensor=torch.randn(3, 4),
    my_other_tensor=torch.ones(3, 4),
    my_metadata="example",
    my_number=123
)

print(data_instance.my_tensor)
print(data_instance.my_metadata)
```

### 3. Call `super().__init__`

It is **crucial** to call `super().__init__(shape, device)` in your subclass's `__init__` method. This initializes the underlying `TensorContainer` and sets up the necessary `shape` and `device` properties for your `TensorAnnotated` instance. The `shape` and `device` should typically be derived from one of your primary tensor attributes.

```python
class MyCustomData(TensorAnnotated):
    my_tensor: Tensor

    def __init__(self, my_tensor: Tensor):
        self.my_tensor = my_tensor
        # Correct way to call super().__init__
        super().__init__(shape=my_tensor.shape, device=my_tensor.device)
```

### Example with PyTree Operations

Once instantiated, your `TensorAnnotated` object will behave like a PyTree, allowing operations like `copy()`, `to()`, `cuda()`, etc.

```python
import torch
from tensorcontainer.tensor_annotated import TensorAnnotated

class MyModelOutput(TensorAnnotated):
    logits: Tensor
    hidden_state: Tensor
    model_name: str

    def __init__(self, logits: Tensor, hidden_state: Tensor, model_name: str):
        self.logits = logits
        self.hidden_state = hidden_state
        self.model_name = model_name
        super().__init__(shape=logits.shape, device=logits.device)

# Create an instance
output = MyModelOutput(
    logits=torch.randn(10, 5),
    hidden_state=torch.randn(10, 128),
    model_name="Transformer"
)

print(f"Original device: {output.logits.device}")
print(f"Original model name: {output.model_name}")

# Move to CPU (if on GPU) or GPU (if on CPU)
new_device = "cpu" if output.logits.is_cuda else "cuda"
output_on_new_device = output.to(new_device)

print(f"New device: {output_on_new_device.logits.device}")
print(f"New model name: {output_on_new_device.model_name}") # Metadata is preserved

# Create a copy
output_copy = output.copy()
print(f"Copy logits are same object? {output_copy.logits is output.logits}") # False, it's a deep copy
print(f"Copy model name: {output_copy.model_name}")
```

## Caveats and Limitations

Understanding these points is crucial for effectively using `TensorAnnotated` and avoiding unexpected behavior:

### 1. Only Annotated Tensors are PyTree Leaves

`TensorAnnotated`'s PyTree integration (flattening and unflattening) *only* considers attributes that are explicitly type-annotated as `torch.Tensor` or `TensorContainer`.

### 2. Attributes from Non-`TensorAnnotated` Parents are Ignored

If your class inherits from a parent class that does *not* subclass `TensorAnnotated`, any attributes defined in that non-`TensorAnnotated` parent will *not* be included in the PyTree operations. They will effectively be lost if you perform operations like `copy()`, `to()`, or `cuda()` on your `TensorAnnotated` instance.

**Example:**

```python
import torch
from torch import Tensor
from tensorcontainer.tensor_annotated import TensorAnnotated

class NonTensorAnnotatedParent:
    def __init__(self, non_pytree_attr: str):
        self.non_pytree_attr = non_pytree_attr

class MyCombinedData(TensorAnnotated, NonTensorAnnotatedParent):
    my_tensor: Tensor

    def __init__(self, my_tensor: Tensor, non_pytree_attr: str):
        self.my_tensor = my_tensor
        NonTensorAnnotatedParent.__init__(self, non_pytree_attr) # Call parent's init
        super().__init__(shape=my_tensor.shape, device=my_tensor.device)

data = MyCombinedData(my_tensor=torch.randn(2), non_pytree_attr="I will be lost")
print(f"Original non_pytree_attr: {data.non_pytree_attr}")

copied_data = data.copy()

# This will raise an AttributeError because non_pytree_attr was not part of the PyTree
try:
    print(f"Copied non_pytree_attr: {copied_data.non_pytree_attr}")
except AttributeError as e:
    print(f"Error accessing copied non_pytree_attr: {e}")
```

### 3. Non-Annotated Attributes are Ignored

Any attribute assigned to `self` within your subclass's `__init__` or other methods that is *not* explicitly type-annotated will also be ignored by the PyTree mechanism. This means they will not be preserved across `copy()`, `to()`, etc.

```python
import torch
from torch import Tensor
from tensorcontainer.tensor_annotated import TensorAnnotated

class MyDataWithNonAnnotated(TensorAnnotated):
    annotated_tensor: Tensor
    # non_annotated_value: int  <-- Missing annotation

    def __init__(self, annotated_tensor: Tensor, non_annotated_value: int):
        self.annotated_tensor = annotated_tensor
        self.non_annotated_value = non_annotated_value # This attribute is not annotated
        super().__init__(shape=annotated_tensor.shape, device=annotated_tensor.device)

data = MyDataWithNonAnnotated(annotated_tensor=torch.randn(2), non_annotated_value=10)
print(f"Original non_annotated_value: {data.non_annotated_value}")

copied_data = data.copy()

# This will raise an AttributeError
try:
    print(f"Copied non_annotated_value: {copied_data.non_annotated_value}")
except AttributeError as e:
    print(f"Error accessing copied non_annotated_value: {e}")
```

### 4. Importance of Calling `super().__init__`

Failing to call `super().__init__(shape, device)` will result in an improperly initialized `TensorAnnotated` instance. Essential properties like `shape` and `device` will not be set, and PyTree operations will likely fail or produce incorrect results.

### 5. Reserved Attributes: `shape` and `device`

The attributes `shape` and `device` are internally managed by `TensorAnnotated` (inherited from `TensorContainer`). You **cannot** define these as annotated attributes in your subclasses. Attempting to do so will result in a `TypeError`.

```python
# This will raise a TypeError
# class InvalidData(TensorAnnotated):
#     shape: torch.Size # ERROR: Cannot define reserved fields
#     my_tensor: Tensor
#
#     def __init__(self, my_tensor: Tensor):
#         self.my_tensor = my_tensor
#         super().__init__(shape=my_tensor.shape, device=my_tensor.device)
```

### 6. Inheritance with Multiple Parents

When using multiple inheritance, `TensorAnnotated` correctly collects annotations from all `TensorAnnotated` parent classes in the Method Resolution Order (MRO). However, ensure that your `__init__` method correctly calls the `__init__` of all relevant parent classes, especially the `TensorAnnotated` ones, passing `shape` and `device` appropriately.

```python
import torch
from torch import Tensor
from tensorcontainer.tensor_annotated import TensorAnnotated

class ParentA(TensorAnnotated):
    tensor_a: Tensor
    def __init__(self, tensor_a: Tensor, **kwargs):
        self.tensor_a = tensor_a
        super().__init__(**kwargs) # Pass kwargs to allow shape/device from child

class ParentB(TensorAnnotated):
    tensor_b: Tensor
    def __init__(self, tensor_b: Tensor, **kwargs):
        self.tensor_b = tensor_b
        super().__init__(**kwargs) # Pass kwargs to allow shape/device from child

class Child(ParentA, ParentB):
    tensor_c: Tensor
    def __init__(self, tensor_a: Tensor, tensor_b: Tensor, tensor_c: Tensor):
        self.tensor_c = tensor_c
        # Call parents' inits, ensuring shape and device are passed to the ultimate TensorAnnotated init
        super().__init__(
            tensor_a=tensor_a,
            tensor_b=tensor_b,
            shape=tensor_c.shape, # Use one of the tensors for shape/device
            device=tensor_c.device
        )

data = Child(torch.randn(5), torch.randn(5), torch.randn(5))
copied_data = data.copy()

assert copied_data.tensor_a is data.tensor_a
assert copied_data.tensor_b is data.tensor_b
assert copied_data.tensor_c is data.tensor_c