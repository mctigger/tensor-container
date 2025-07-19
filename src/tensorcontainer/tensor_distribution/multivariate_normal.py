from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from .base import TensorDistribution


class TensorMultivariateNormal(TensorDistribution):
    _loc: Tensor
    _covariance_matrix: Tensor | None = None
    _precision_matrix: Tensor | None = None
    _scale_tril: Tensor | None = None

    def __init__(
        self,
        loc: Tensor,
        covariance_matrix: Tensor | None = None,
        precision_matrix: Tensor | None = None,
        scale_tril: Tensor | None = None,
    ):
        num_params = sum(
            p is not None for p in [covariance_matrix, precision_matrix, scale_tril]
        )
        if num_params != 1:
            raise ValueError(
                "Expected exactly one of `covariance_matrix`, `precision_matrix`, "
                f"`scale_tril` to be specified, but got {num_params}."
            )

        self._loc = loc
        self._covariance_matrix = covariance_matrix
        self._precision_matrix = precision_matrix
        self._scale_tril = scale_tril
        super().__init__(loc.shape, loc.device)  # Call super().__init__ here

    def dist(self) -> MultivariateNormal:
        return MultivariateNormal(
            loc=self._loc,
            covariance_matrix=self._covariance_matrix,
            precision_matrix=self._precision_matrix,
            scale_tril=self._scale_tril,
        )

    @classmethod
    def _unflatten_distribution(
        cls, attributes: Dict[str, Any]
    ) -> TensorMultivariateNormal:
        instance = cls(
            loc=attributes["_loc"],
            covariance_matrix=attributes.get("_covariance_matrix"),
            precision_matrix=attributes.get("_precision_matrix"),
            scale_tril=attributes.get("_scale_tril"),
        )
        return instance

    @property
    def loc(self) -> Tensor:
        return self._loc

    @property
    def covariance_matrix(self) -> Tensor:
        return self.dist().covariance_matrix

    @property
    def precision_matrix(self) -> Tensor:
        return self.dist().precision_matrix

    @property
    def scale_tril(self) -> Tensor:
        return self.dist().scale_tril

    @property
    def batch_shape(self) -> torch.Size:
        """Returns the batch shape of the distribution."""
        return self._loc.shape[:-1]

    @property
    def event_shape(self) -> torch.Size:
        """Returns the event shape of the distribution."""
        return self._loc.shape[-1:]
