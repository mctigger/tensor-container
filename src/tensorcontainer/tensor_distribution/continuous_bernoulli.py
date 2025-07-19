from typing import Optional, Tuple, Union, get_args

import torch
from torch import Tensor
from torch.distributions import ContinuousBernoulli as TorchContinuousBernoulli
from torch.types import Number

from .base import TensorDistribution


class TensorContinuousBernoulli(TensorDistribution):
    _probs: Optional[Tensor]
    _logits: Optional[Tensor]
    _lims: Tuple[float, float]

    def __init__(
        self,
        probs: Optional[Union[Tensor, Number]] = None,
        logits: Optional[Union[Tensor, Number]] = None,
        lims: Tuple[float, float] = (0.499, 0.501),
    ) -> None:
        self._lims = lims

        if probs is not None and logits is not None:
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        elif probs is None and logits is None:
            raise ValueError("Either `probs` or `logits` must be specified.")

        if probs is not None and isinstance(probs, get_args(Number)):
            self._probs = torch.tensor(probs)
        else:
            self._probs = probs

        if logits is not None and isinstance(logits, get_args(Number)):
            self._logits = torch.tensor(logits)
        else:
            self._logits = logits

        if self._probs is not None:
            batch_shape = self._probs.shape
            device = self._probs.device
        elif self._logits is not None:
            batch_shape = self._logits.shape
            device = self._logits.device
        else:
            # This case should ideally not be reached due to the checks above,
            # but as a fallback for type inference or future changes.
            raise ValueError("Either `probs` or `logits` must be specified.")

        super().__init__(shape=batch_shape, device=device)

    def dist(self) -> TorchContinuousBernoulli:
        return TorchContinuousBernoulli(
            probs=self._probs,
            logits=self._logits,
            lims=self._lims,
        )

    @classmethod
    def _unflatten_distribution(
        cls,
        attributes: dict,
    ) -> "TensorContinuousBernoulli":
        return cls(
            probs=attributes.get("_probs"),
            logits=attributes.get("_logits"),
            lims=attributes["_lims"],
        )

    @property
    def probs(self) -> Tensor:
        return self.dist().probs

    @property
    def logits(self) -> Tensor:
        return self.dist().logits

    @property
    def mean(self) -> Tensor:
        return self.dist().mean

    @property
    def variance(self) -> Tensor:
        return self.dist().variance

    @property
    def param_shape(self) -> torch.Size:
        return self.dist().param_shape
