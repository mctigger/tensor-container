import torch
from torch.distributions import Independent, Normal

from tensorcontainer.tensor_distribution import TensorNormal


def test_rsample_returns_differentiable_tensor_and_correct_shape():
    # scalar Normal distribution
    loc = torch.tensor(0.0, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=True)
    td = TensorNormal(
        loc=loc,
        scale=scale,
        reinterpreted_batch_ndims=0,
        shape=loc.shape,
        device=loc.device,
    )

    # default rsample: no sample_shape
    x = td.rsample()
    # must require grad because rsample is reparameterized
    assert isinstance(x, torch.Tensor)
    assert x.requires_grad
    assert x.shape == torch.Size([])

    # with sample_shape
    x2 = td.rsample(sample_shape=torch.Size([5]))
    assert x2.shape == torch.Size([5])
    assert x2.requires_grad


def test_sample_returns_nondifferentiable_tensor_and_correct_shape():
    loc = torch.zeros(3)
    scale = torch.ones(3)
    td = TensorNormal(
        loc=loc,
        scale=scale,
        reinterpreted_batch_ndims=1,
        shape=loc.shape,
        device=loc.device,
    )

    # default sample: draws one event of shape (3,)
    s = td.sample()
    assert isinstance(s, torch.Tensor)
    # sample is not reparameterized => no grad
    assert not s.requires_grad
    assert s.shape == torch.Size([3])

    # with sample_shape
    s2 = td.sample(sample_shape=torch.Size([4, 2]))
    # event_shape=(3,), so shape = (4,2,3)
    assert s2.shape == torch.Size([4, 2, 3])


def test_mean_stddev_mode_match_underlying_distribution():
    loc = torch.linspace(-1, 1, steps=4)
    scale = torch.linspace(0.5, 1.5, steps=4)
    td = TensorNormal(
        loc=loc,
        scale=scale,
        reinterpreted_batch_ndims=1,
        shape=loc.shape,
        device=loc.device,
    )

    dist = Independent(Normal(loc=loc, scale=scale), 1)
    # properties on TensorNormal
    assert torch.allclose(td.mean, dist.mean)
    assert torch.allclose(td.stddev, dist.stddev)
    # mode for Normal is the same as mean
    assert torch.allclose(td.mode, dist.mode)


def test_entropy_matches_underlying_distribution():
    loc = torch.zeros(2, 3)
    scale = torch.ones(2, 3) * 2.0
    # treat last two dims as event dims => entropy summed over them
    td = TensorNormal(
        loc=loc,
        scale=scale,
        reinterpreted_batch_ndims=2,
        shape=loc.shape,
        device=loc.device,
    )

    dist = Independent(Normal(loc=loc, scale=scale), 2)
    ent_td = td.entropy()
    ent_dist = dist.entropy()
    # shapes should match batch_shape=()
    assert ent_td.shape == ent_dist.shape
    assert torch.allclose(ent_td, ent_dist)


def test_log_prob_agrees_with_underlying_distribution():
    loc = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    td = TensorNormal(
        loc=loc,
        scale=scale,
        reinterpreted_batch_ndims=2,
        shape=loc.shape,
        device=loc.device,
    )

    dist = Independent(Normal(loc=loc, scale=scale), 2)
    # draw a sample to evaluate log_prob
    sample = td.sample(sample_shape=torch.Size([5]))  # shape (5,2,2)
    lp_td = td.log_prob(sample)
    lp_dist = dist.log_prob(sample)
    # log_prob should match exactly, shape = (5,)
    assert lp_td.shape == lp_dist.shape
    assert torch.allclose(lp_td, lp_dist)
