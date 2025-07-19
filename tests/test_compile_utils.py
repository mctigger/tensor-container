import pytest
import torch
import torch._dynamo

from tests.compile_utils import (
    get_graph_breaks_and_recompiles,
    run_and_compare_compiled,
    run_and_count_graph_breaks,
    run_and_count_recompiles,
)
from tests.conftest import skipif_no_compile


def no_break(x):
    # This operation should not cause a graph break
    return x[0] + 1


def one_break(x):
    # This operation is known to cause a graph break
    torch._dynamo.graph_break()
    return x[0] + 1


def two_breaks(x):
    # These operations are known to cause graph breaks
    tensor = x[0]
    torch._dynamo.graph_break()
    torch._dynamo.graph_break()
    return tensor[0] + 1


def three_breaks(x):
    # These operations are known to cause graph breaks
    tensor = x[0]
    torch._dynamo.graph_break()
    torch._dynamo.graph_break()
    torch._dynamo.graph_break()
    return tensor[0] + 1


@skipif_no_compile
def test_run_and_compare_compiled_fails():
    # Assert that this raises an AssertionError.
    x = torch.randn(3, 4)
    with pytest.raises(AssertionError):
        # The 'one_break' function causes 1 graph break, but we expect 0.
        # This should trigger an AssertionError.
        run_and_compare_compiled(
            one_break, (x,), expected_graph_breaks=0, fullgraph=False
        )


def recursive_recompile(i):
    if i > 0:
        i -= 1
        return recursive_recompile(i)
    else:
        return torch.tensor([1])


def one_break_one_recompile_func(i):
    if i == 1:
        torch._dynamo.graph_break()
    return torch.tensor([i])


@pytest.mark.parametrize(
    "func, expected_graph_breaks, fullgraph",
    [
        (no_break, 0, True),
        (one_break, 1, False),
        (two_breaks, 2, False),
        (three_breaks, 3, False),
    ],
)
@skipif_no_compile
def test_run_and_compare_compiled_parameterized(func, expected_graph_breaks, fullgraph):
    """Tests that a function has the expected number of graph breaks."""
    x = torch.randn(3, 4)
    run_and_compare_compiled(
        func, (x,), expected_graph_breaks=expected_graph_breaks, fullgraph=fullgraph
    )


@pytest.mark.parametrize(
    "func, expected_graph_breaks, fullgraph",
    [
        (no_break, 0, True),
        (one_break, 1, False),
        (two_breaks, 2, False),
        (three_breaks, 3, False),
    ],
)
@skipif_no_compile
def test_run_and_count_graph_breaks_parameterized(
    func, expected_graph_breaks, fullgraph
):
    """Tests that a function has the expected number of graph breaks."""
    x = torch.randn(3, 4)
    run_and_count_graph_breaks(
        func, x, expected_graph_breaks=expected_graph_breaks, fullgraph=fullgraph
    )


@pytest.mark.parametrize(
    "func, args, expected_recompiles",
    [
        (recursive_recompile, ((1,),), 0),
        (recursive_recompile, ((1,), (3,)), 1),
    ],
)
@skipif_no_compile
def test_run_and_count_recompiles_parameterized(func, args, expected_recompiles):
    """Tests that a stateful functor recompiles as expected."""
    run_and_count_recompiles(func, *args, expected_recompiles=expected_recompiles)


@pytest.mark.parametrize(
    "func, args, expected_breaks, expected_recompiles, fullgraph",
    [
        (no_break, ((torch.randn(2),),), 0, 0, True),
        (one_break, ((torch.randn(2),),), 1, 0, False),
        (two_breaks, ((torch.randn(2, 2),),), 2, 0, False),
        (recursive_recompile, ((1,), (2,)), 0, 1, True),
        (recursive_recompile, ((1,), (2,), (3,)), 0, 2, True),
        (one_break_one_recompile_func, ((1,), (2,)), 1, 1, False),
    ],
)
@skipif_no_compile
def test_get_graph_breaks_and_recompiles_parameterized(
    func, args, expected_breaks, expected_recompiles, fullgraph
):
    """Tests that a function has the expected number of graph breaks and recompiles."""
    breaks, recompiles = get_graph_breaks_and_recompiles(
        func, *args, fullgraph=fullgraph
    )
    assert breaks == expected_breaks
    assert recompiles == expected_recompiles
