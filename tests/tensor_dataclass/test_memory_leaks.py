import dataclasses
import gc
import weakref

import torch

from tensorcontainer import TensorDataClass


@dataclasses.dataclass
class MyData(TensorDataClass):
    my_tensor: torch.Tensor


def test_tensor_dataclass_memory_leak():
    # Instantiate MyData with a sample tensor
    data_instance = MyData(
        my_tensor=torch.tensor([1.0, 2.0, 3.0], device="cpu"),
        shape=(3,),
        device=torch.device("cpu"),
    )

    # Extract the tensor from the MyData instance into a separate variable
    extracted_tensor = data_instance.my_tensor

    # Create a weakref.ref to the MyData instance
    weak_ref_to_instance = weakref.ref(data_instance)

    # Delete the MyData instance
    del data_instance

    # Call gc.collect() to force garbage collection
    gc.collect()

    # Assert that the weak reference is now None, indicating the instance was collected
    assert weak_ref_to_instance() is None

    # Assert that the extracted tensor is not None to ensure it still exists
    assert extracted_tensor is not None
