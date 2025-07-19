# Testing Guide
Our testing philosophy is simple: **every piece of functionality should be verified by a clean, isolated, and readable test**. You MUST use the `pytest` framework to write and run tests. The following are core principles for creating a high-quality and maintainable test suite that you MUST adhere to.

-----

### Core Principles

#### Test File Structure

  - Every test file (e.g., `test_container.py`) MUST begin with a module-level docstring.
  - This docstring MUST provide a short summary of the test classes contained within that file, giving a high-level overview of the test coverage.

#### Test Class Structure

  - All tests MUST be organized into classes.

  - Every test class MUST have a docstring that clearly explains its purpose. The docstring MUST follow this format:

    1.  **Summary**: 1-2 sentences giving a high-level overview of the feature being tested.
    2.  **Test List**: A bulleted list detailing the specific behaviors verified by the test suite.

  - **Example**:

    ```python
    class TestTensorContainerInitialization:
        """
        Tests the initialization logic of the TensorContainer.

        This suite verifies that:
        - The container is created successfully with a valid tensor.
        - The 'requires_grad' flag is set correctly during instantiation.
        - A TypeError is raised when input data is not a PyTorch tensor.
        """
        # ... test methods go here
    ```

#### Test Method Rules

  - **Isolate Everything**: Each test method (e.g., `def test_*()`) should verify only ONE specific behavior or outcome. Do NOT test multiple things at once.
  - **Parametrize, Don't Branch**: You MUST use `pytest.mark.parametrize` to test multiple scenarios of a single behavior. This helps avoid `if/else` logic inside the test itself, keeping its purpose clear and focused.
      - When parametrizing a test, make sure that all the parameters are of a similar type.
      - **Good Parametrization**:
        ```python
        @pytest.mark.parametrize(
                "original_shape,expected_elements",
                [
                    ((2, 3), 6),
                    ((4, 5), 20),
                    ((1, 10), 10),
                    ((3, 2, 2), 12),
                ],
            )
        ```
      - **Bad Parametrization**:
        ```python
        @pytest.mark.parametrize(
                "shape_or_type,expected_elements_or_type",
                [
                    ((2, 3), 6),
                    (torch.float, torch.float),
                    (torch.double, torch.double),
                    ((3, 2, 2), 12),
                ],
            )
        ```
  - **Clear Naming**: Test classes and methods MUST have descriptive names that immediately convey what they are testing.
  - **Do Not Overcomplicate**: Instead of creating complex tests that are very general, create many simple test cases that check exactly one thing.
  - **Assert Only Relevant Properties**: Complicated asserts indicate an overly complex test. In that case, split the test into multiple, simpler tests.
  - **Add Docstrings**: Add a small docstring that explains the reasoning behind the test.
    ```python
    def test_optional_field_none(self):
        """
        The TensorDataClass should allow fields to be optional.
        These fields are set later.
        """
    ```
  - **Use Testing Utilities**:
      - You MUST use the `context7 MCP` when you need to access the API of installed packages (such as `torch`, `numpy`, `torchvision`, etc.).
      - You MUST use the testing utilities from `torch.testing` when working with tensors.
  - **Do Not Modify Implementation**: You are NEVER allowed to change the implementation that is being tested. For example, if your task is to test `TensorContainer.flatten()`, you are not allowed to make any changes in `TensorContainer`.

-----

### Testing Considerations for torch.compile

When testing code that uses [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html), be aware of global state issues that can affect test isolation:

- **Validation State Leakage**: [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) can disable [`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) validation globally, causing subsequent tests to behave differently than expected.
- **Test Isolation**: Use the [`preserve_distributions_validation`](tests/tensor_distribution/conftest.py:7) fixture (automatically applied) to ensure validation state doesn't leak between tests.
- **Explicit Validation**: Use the [`with_distributions_validation`](tests/tensor_distribution/conftest.py:25) fixture when you need to guarantee validation is enabled for specific test scenarios.

These fixtures prevent hard-to-debug test failures where validation-dependent tests pass or fail based on the order of test execution.