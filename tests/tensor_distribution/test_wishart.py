import pytest
import torch
from _pytest.fixtures import FixtureRequest
from torch.distributions import Wishart as TorchWishart

from tensorcontainer.tensor_distribution.wishart import TensorWishart
from tests.conftest import skipif_no_compile
from tests.tensor_distribution.conftest import (
    assert_property_values_match,
    compile_args,
)


@pytest.fixture(
    params=[
        (torch.tensor([2.0]), torch.eye(2)),  # df, covariance_matrix
        (torch.tensor([3.0]), torch.eye(3)),
        (torch.tensor([4.0]), torch.eye(2)),
    ]
)
def wishart_params(request: FixtureRequest):
    df, matrix = request.param
    return df, matrix


class TestTensorWishart:
    def test_init_covariance_matrix(self, wishart_params):
        df, matrix = wishart_params
        dist = TensorWishart(df=df, covariance_matrix=matrix)
        torch_dist = TorchWishart(df=df, covariance_matrix=matrix)
        assert isinstance(dist.dist(), TorchWishart)
        assert torch.equal(dist.df, df)
        assert torch.equal(dist.covariance_matrix, torch_dist.covariance_matrix)

    def test_init_precision_matrix(self, wishart_params):
        df, matrix = wishart_params
        precision_matrix = torch.inverse(matrix)
        dist = TensorWishart(df=df, precision_matrix=precision_matrix)
        torch_dist = TorchWishart(df=df, precision_matrix=precision_matrix)
        assert isinstance(dist.dist(), TorchWishart)
        assert torch.equal(dist.df, df)
        assert torch.allclose(dist.precision_matrix, torch_dist.precision_matrix)

    def test_init_scale_tril(self, wishart_params):
        df, matrix = wishart_params
        scale_tril = torch.linalg.cholesky(matrix)
        dist = TensorWishart(df=df, scale_tril=scale_tril)
        torch_dist = TorchWishart(df=df, scale_tril=scale_tril)
        assert isinstance(dist.dist(), TorchWishart)
        assert torch.equal(dist.df, df)
        assert torch.allclose(dist.scale_tril, torch_dist.scale_tril)

    def test_property_values_match(self, wishart_params):
        df, matrix = wishart_params
        tdist = TensorWishart(df=df, covariance_matrix=matrix)
        assert_property_values_match(tdist)

    def test_properties(self, wishart_params):
        df, matrix = wishart_params
        dist = TensorWishart(df=df, covariance_matrix=matrix)
        torch_dist = TorchWishart(df=df, covariance_matrix=matrix)

        assert torch.equal(dist.covariance_matrix, torch_dist.covariance_matrix)
        assert torch.allclose(dist.precision_matrix, torch_dist.precision_matrix)
        assert torch.allclose(dist.scale_tril, torch_dist.scale_tril)

    @skipif_no_compile
    @pytest.mark.parametrize("compile_args", compile_args)
    def test_compile(self, wishart_params, compile_args):
        # Mark expected failures for fullgraph=True with dynamic=True/False
        # due to data-dependent branching in PyTorch Wishart implementation
        if compile_args.get("fullgraph") and compile_args.get("dynamic") in [
            True,
            False,
        ]:
            pytest.xfail(
                "PyTorch Wishart has data-dependent branching incompatible with fullgraph=True"
            )
        df, matrix = wishart_params
        dist = TensorWishart(df=df, covariance_matrix=matrix)

        # Compile individual methods
        compiled_sample = torch.compile(dist.sample, **compile_args)
        compiled_log_prob = torch.compile(dist.log_prob, **compile_args)
        compiled_mean = torch.compile(lambda: dist.mean, **compile_args)
        compiled_variance = torch.compile(lambda: dist.variance, **compile_args)

        # Test basic methods after compilation
        sample = compiled_sample()
        assert sample.shape == dist.sample().shape

        # Create a dummy value for log_prob
        value_shape = sample.shape
        value = torch.randn(value_shape, dtype=dist.dtype, device=dist.device)
        log_prob = compiled_log_prob(value)
        assert log_prob.shape == dist.log_prob(value).shape

        mean = compiled_mean()
        assert mean.shape == dist.mean.shape

        variance = compiled_variance()
        assert variance.shape == dist.variance.shape
