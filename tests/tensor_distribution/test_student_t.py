import pytest
import torch
from torch.distributions import StudentT

from tensorcontainer.tensor_distribution.student_t import TensorStudentT
from tests.compile_utils import run_and_compare_compiled


@pytest.mark.parametrize("df_val", [1.0, torch.tensor(2.0)])
@pytest.mark.parametrize("loc_val", [0.0, torch.tensor(1.0)])
@pytest.mark.parametrize("scale_val", [1.0, torch.tensor(2.0)])
def test_student_t_init(df_val, loc_val, scale_val):
    # Determine the expected batch_shape based on the input parameters
    df_tensor = torch.as_tensor(df_val)
    loc_tensor = torch.as_tensor(
        loc_val, dtype=df_tensor.dtype, device=df_tensor.device
    )
    scale_tensor = torch.as_tensor(
        scale_val, dtype=df_tensor.dtype, device=df_tensor.device
    )
    expected_batch_shape = torch.broadcast_shapes(
        df_tensor.shape, loc_tensor.shape, scale_tensor.shape
    )

    dist = TensorStudentT(df_val, loc_val, scale_val)
    assert dist.batch_shape == expected_batch_shape
    assert isinstance(dist.df, torch.Tensor)
    assert isinstance(dist.loc, torch.Tensor)
    assert isinstance(dist.scale, torch.Tensor)

    # Test properties
    assert torch.allclose(dist.df, df_tensor)
    assert torch.allclose(dist.loc, loc_tensor)
    assert torch.allclose(dist.scale, scale_tensor)

    # Test dist() method
    torch_dist = StudentT(df=df_tensor, loc=loc_tensor, scale=scale_tensor)
    assert torch.allclose(
        dist.log_prob(torch.tensor(0.5)), torch_dist.log_prob(torch.tensor(0.5))
    )
    assert torch.allclose(dist.mean, torch_dist.mean, equal_nan=True)
    assert torch.allclose(dist.variance, torch_dist.variance, equal_nan=True)
    assert torch.allclose(dist.stddev, torch_dist.stddev, equal_nan=True)

    # Test sample and rsample
    sample = dist.sample()
    assert sample.shape == expected_batch_shape
    rsample = dist.rsample()
    assert rsample.shape == expected_batch_shape


@pytest.mark.parametrize("df", [1.0, torch.tensor(2.0)])
@pytest.mark.parametrize("loc", [0.0, torch.tensor(1.0)])
@pytest.mark.parametrize("scale", [1.0, torch.tensor(2.0)])
def test_student_t_compile(df, loc, scale):
    dist = TensorStudentT(df, loc, scale)

    def get_sample(td):
        return td.sample()

    def get_rsample(td):
        return td.rsample()

    def get_log_prob(td):
        return td.log_prob(torch.tensor(0.5))

    def get_mean(td):
        return td.mean

    def get_variance(td):
        return td.variance

    def get_stddev(td):
        return td.stddev

    run_and_compare_compiled(get_sample, dist, fullgraph=False)
    run_and_compare_compiled(get_rsample, dist, fullgraph=False)
    run_and_compare_compiled(get_log_prob, dist, fullgraph=False)
    run_and_compare_compiled(get_mean, dist, fullgraph=False)
    run_and_compare_compiled(get_variance, dist, fullgraph=False)
    run_and_compare_compiled(get_stddev, dist, fullgraph=False)
