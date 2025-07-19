"""Tests for mathematical operations on TensorDict."""

import pytest
import torch

from tensorcontainer.tensor_dict import TensorDict
from tests.conftest import skipif_no_compile

from .common import compare_nested_dict


class TestTensorDictMathOps:
    """
    Tests the mathematical operations of the TensorDict.

    This suite verifies that:
    - Basic arithmetic operations (add, sub, mul, div) are applied correctly.
    - Unary operations (abs, sqrt, log, neg) produce the expected results.
    - Other mathematical functions (pow, clamp) work as intended.
    - All operations are compatible with torch.compile.
    - The operations return a new TensorDict with the correct shape and device.
    """

    def _test_op(self, td, op_func, expected_op, compile_op):
        """
        Helper function to test a mathematical operation on a TensorDict.

        Args:
            td (TensorDict): The TensorDict instance to test.
            op_func (callable): The function to apply to the TensorDict.
            expected_op (callable): The corresponding torch function for verification.
            compile_op (bool): Whether to compile the operation with torch.compile.
        """
        if compile_op:
            op_func = torch.compile(op_func, fullgraph=True)

        result_td = op_func(td)

        assert isinstance(result_td, TensorDict)
        assert result_td is not td
        assert result_td.shape == td.shape
        assert result_td.device == td.device

        compare_nested_dict(td.data, result_td.data, expected_op)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_abs(self, nested_dict, compile_op):
        """Tests the element-wise absolute value operation."""
        shape = (2, 2)
        data = nested_dict(shape)

        # Negate some values to test abs
        data["x"]["a"] = -data["x"]["a"]
        data["y"] = -data["y"]

        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.abs()

        self._test_op(td, op, torch.abs, compile_op)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_add(self, nested_dict, compile_op):
        """Tests the element-wise addition operation."""
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.add(2)

        self._test_op(td, op, lambda x: x + 2, compile_op)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_sub(self, nested_dict, compile_op):
        """Tests the element-wise subtraction operation."""
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.sub(2)

        self._test_op(td, op, lambda x: x - 2, compile_op)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_mul(self, nested_dict, compile_op):
        """Tests the element-wise multiplication operation."""
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.mul(2)

        self._test_op(td, op, lambda x: x * 2, compile_op)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_div(self, nested_dict, compile_op):
        """Tests the element-wise division operation."""
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.div(2)

        self._test_op(td, op, lambda x: x / 2, compile_op)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_pow(self, nested_dict, compile_op):
        """Tests the element-wise power operation."""
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.pow(2)

        self._test_op(td, op, lambda x: x**2, compile_op)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_sqrt(self, nested_dict, compile_op):
        """Tests the element-wise square root operation."""
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.sqrt()

        self._test_op(td, op, torch.sqrt, compile_op)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_log(self, nested_dict, compile_op):
        """Tests the element-wise natural logarithm operation."""
        shape = (2, 2)
        data = nested_dict(shape)
        # Ensure values are positive for log
        for k, v in data.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    data[k][sub_k] = sub_v.abs() + 1
            else:
                data[k] = v.abs() + 1
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.log()

        self._test_op(td, op, torch.log, compile_op)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_neg(self, nested_dict, compile_op):
        """Tests the element-wise negation operation."""
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu")

        def op(t):
            return t.neg()

        self._test_op(td, op, lambda x: -x, compile_op)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_op", [True, False])
    def test_clamp(self, nested_dict, compile_op):
        """Tests the element-wise clamp operation."""
        shape = (2, 2)
        data = nested_dict(shape)
        td = TensorDict(data, shape, device="cpu").to(torch.float32)

        def op(t):
            return t.clamp(min=0.2, max=0.8)

        self._test_op(td, op, lambda x: torch.clamp(x, min=0.2, max=0.8), compile_op)
