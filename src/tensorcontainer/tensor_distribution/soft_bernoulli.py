from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Independent

from tensorcontainer.distributions.soft_bernoulli import SoftBernoulli

from .base import TensorDistribution


class TensorSoftBernoulli(TensorDistribution):
    _probs: Optional[Tensor] = None
    _logits: Optional[Tensor] = None
    _validate_args: Optional[bool] = None

    def __init__(
        self,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args: Optional[bool] = None,
    ):
        self._validate_args = validate_args
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )

        if probs is not None:
            batch_shape = probs.shape
            device = probs.device
        else:
            assert logits is not None
            batch_shape = logits.shape
            device = logits.device

        self._probs = probs
        self._logits = logits
        self._validate_args = validate_args
        super().__init__(shape=batch_shape, device=device)

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]):
        return cls(
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
            validate_args=attributes.get("_validate_args"),
        )

    @property
    def probs(self):
        if self._probs is None:
            assert self._logits is not None
            self._probs = torch.sigmoid(self._logits)
        return self._probs

    @probs.setter
    def probs(self, value):
        self._probs = value
        if value is not None:
            self._logits = None

    @property
    def logits(self):
        if self._logits is None:
            assert self._probs is not None
            self._logits = torch.log(self._probs / (1 - self._probs + 1e-8))
        return self._logits

    @logits.setter
    def logits(self, value):
        self._logits = value
        if value is not None:
            self._probs = None

    @property
    def param_shape(self):
        return self.batch_shape

    def dist(self):
        if self._probs is not None:
            return Independent(
                SoftBernoulli(
                    probs=self._probs,
                    validate_args=self._validate_args,
                ),
                0,
            )
        else:
            return Independent(
                SoftBernoulli(
                    logits=self._logits,
                    validate_args=self._validate_args,
                ),
                0,
            )
