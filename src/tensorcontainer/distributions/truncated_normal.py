from __future__ import annotations

import torch
import torch.distributions
from torch import Tensor


class TruncatedNormal(torch.distributions.Normal):
    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        low: Tensor,
        high: Tensor,
        eps: float = 1e-6,
        validate_args=None,
    ):
        super().__init__(loc, scale, validate_args)
        self.low = low
        self.high = high
        self.eps = eps

    def rsample(self, sample_shape=torch.Size()):
        event = super().rsample(sample_shape)

        event_clamp = torch.clamp(event, self.low + self.eps, self.high - self.eps)

        event = event - event.detach() + event_clamp.detach()

        return event

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
