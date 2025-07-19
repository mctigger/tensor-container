from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from torch import Size, Tensor
from torch._C import device
from torch.distributions import Distribution, kl_divergence, register_kl

from tensorcontainer.tensor_annotated import TensorAnnotated
from tensorcontainer.tensor_dict import TDCompatible


class TensorDistribution(TensorAnnotated):
    """
    Base class for tensor-aware probability distributions that integrate with TensorDict ecosystem.

    TensorDistribution extends TensorAnnotated to provide a unified interface for probability
    distributions that can be seamlessly used with TensorDict operations like batching,
    device movement, and shape transformations.

    Key Design Principles:
    ----------------------
    1. **Annotated Attributes**: Only tensor attributes marked with type annotations get
       automatically transformed by TensorAnnotated operations (e.g., .to(), .expand()).

    2. **Deferred Validation**: The __init__ method should mirror torch.distributions
       equivalents but without validate_args, deferring all validation to the underlying
       torch.distributions.

    3. **Lazy Distribution Creation**: The actual torch.distributions.Distribution instance
       is created on-demand via the dist() method, allowing for efficient tensor operations
       without premature distribution instantiation.

    4. **Reconstruction Pattern**: Uses _unflatten_distribution() to rebuild distributions
       from serialized tensor and metadata attributes, enabling proper deserialization.

    Usage Pattern:
    --------------
    Subclasses should:
    - Annotate all tensor parameters as class attributes (e.g., _probs: Optional[Tensor] = None)
    - Implement __init__ to mirror the corresponding torch.distributions constructor
    - Implement dist() to return the underlying torch.distributions.Distribution
    - Override _unflatten_distribution() if custom reconstruction logic is needed

    Example:
    --------
    ```python
    class TensorNormal(TensorDistribution):
        _loc: Optional[Tensor] = None
        _scale: Optional[Tensor] = None

        def __init__(self, loc: Tensor, scale: Tensor):
            self._loc = loc
            self._scale = scale
            super().__init__(loc.shape, loc.device)

        def dist(self) -> Distribution:
            return Normal(self._loc, self._scale)
    ```
    """

    def __init__(
        self, shape: Size | List[int] | Tuple[int], device: str | device | int | None
    ):
        """
        Initialize the TensorDistribution.

        Args:
            shape: Shape of the distribution's batch dimensions
            device: Device where tensors should be placed

        Note:
            This calls dist() to validate that the distribution can be constructed,
            then initializes the parent TensorAnnotated class.
        """
        # Validate that the distribution can be constructed by calling dist()
        # We pass validate_args=False to the super().__init__ because the
        # TensorDistribution itself handles validation by calling self.dist()
        # and relies on the underlying torch.distributions for parameter validation.
        self.dist()
        super().__init__(shape, device)

    @classmethod
    def _init_from_reconstructed(
        cls,
        tensor_attributes: Dict[str, TDCompatible],
        meta_attributes: Dict[str, Any],
        device,
        shape,
    ):
        """
        Internal method used by TensorAnnotated for reconstructing distributions.

        This method is called during operations like .to(), .expand(), etc. to rebuild
        the distribution instance from its constituent tensor and metadata attributes.

        Args:
            tensor_attributes: Dictionary of annotated tensor attributes
            meta_attributes: Dictionary of non-tensor metadata
            device: Target device (unused, inferred from tensors)
            shape: Target shape (unused, inferred from tensors)

        Returns:
            Reconstructed TensorDistribution instance
        """
        return cls._unflatten_distribution({**tensor_attributes, **meta_attributes})

    @classmethod
    def _unflatten_distribution(cls, attributes: Dict[str, Any]):
        """
        Reconstruct a distribution from flattened tensor and metadata attributes.

        This method should be overridden by subclasses that need custom reconstruction
        logic. The default implementation assumes all attributes can be passed directly
        to the constructor.

        Args:
            attributes: Dictionary mapping attribute names to values

        Returns:
            New instance of the distribution class

        Example:
            For TensorCategorical, this extracts _probs and _logits from attributes
            and passes them to the constructor:
            ```python
            return cls(
                probs=attributes.get("_probs"),
                logits=attributes.get("_logits"),
            )
            ```
        """
        return cls(**attributes)

    @abstractmethod
    def dist(self) -> Distribution:
        """
        Return the underlying torch.distributions.Distribution instance.

        This method should create and return the actual PyTorch distribution object
        using the annotated tensor attributes. It should NOT wrap the distribution
        with Independent or other wrappers - return the raw distribution directly.

        Returns:
            The underlying torch.distributions.Distribution instance

        Example:
            ```python
            def dist(self) -> Distribution:
                return OneHotCategoricalStraightThrough(
                    probs=self._probs, logits=self._logits
                )
            ```
        """

    def rsample(self, sample_shape: Size = Size()) -> Tensor:
        """
        Generate reparameterized samples from the distribution.

        Args:
            sample_shape: Shape of samples to generate

        Returns:
            Reparameterized samples with gradient flow enabled
        """
        return self.dist().rsample(sample_shape)

    def sample(self, sample_shape: Size = Size()) -> Tensor:
        """
        Generate samples from the distribution.

        Args:
            sample_shape: Shape of samples to generate

        Returns:
            Samples from the distribution (may not have gradient flow)
        """
        return self.dist().sample(sample_shape)

    def entropy(self) -> Tensor:
        """
        Compute the entropy of the distribution.

        Returns:
            Entropy tensor with shape matching the distribution's batch shape
        """
        return self.dist().entropy()

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Compute the log probability density/mass of the given value.

        Args:
            value: Value to compute log probability for

        Returns:
            Log probability tensor with shape matching the distribution's batch shape
        """
        return self.dist().log_prob(value)

    @property
    def mean(self) -> Tensor:
        """Mean of the distribution."""
        return self.dist().mean

    @property
    def stddev(self) -> Tensor:
        """Standard deviation of the distribution."""
        return self.dist().stddev

    @property
    def mode(self) -> Tensor:
        """Mode of the distribution."""
        return self.dist().mode

    @property
    def variance(self) -> Tensor:
        """Variance of the distribution."""
        return self.dist().variance

    @property
    def batch_shape(self) -> Size:
        """Batch shape of the distribution."""
        return self.dist().batch_shape

    @property
    def event_shape(self) -> Size:
        """Event shape of the distribution."""
        return self.dist().event_shape

    @property
    def support(self):
        """Support of the distribution."""
        return self.dist().support

    @property
    def has_rsample(self) -> bool:
        """Whether the distribution supports reparameterized sampling."""
        return self.dist().has_rsample

    @property
    def has_enumerate_support(self) -> bool:
        """Whether the distribution supports enumeration over its support."""
        return self.dist().has_enumerate_support

    @property
    def arg_constraints(self):
        """Argument constraints for the distribution parameters."""
        return self.dist().arg_constraints

    def cdf(self, value: Tensor) -> Tensor:
        """
        Compute the cumulative distribution function at the given value.

        Args:
            value: Value to compute CDF for

        Returns:
            CDF tensor with shape matching the distribution's batch shape
        """
        return self.dist().cdf(value)

    def icdf(self, value: Tensor) -> Tensor:
        """
        Compute the inverse cumulative distribution function at the given value.

        Args:
            value: Value to compute inverse CDF for

        Returns:
            Inverse CDF tensor with shape matching the distribution's batch shape
        """
        return self.dist().icdf(value)

    def enumerate_support(self, expand: bool = True) -> Tensor:
        """
        Enumerate over all possible values in the distribution's support.

        Args:
            expand: Whether to expand the support over batch dimensions

        Returns:
            Tensor containing all possible values in the support
        """
        return self.dist().enumerate_support(expand)

    def perplexity(self) -> Tensor:
        """
        Compute the perplexity of the distribution.

        Returns:
            Perplexity tensor with shape matching the distribution's batch shape
        """
        return self.dist().perplexity()

    def _extended_shape(self, sample_shape: Size = Size()) -> Size:
        """
        Compute the extended shape for sampling.

        Args:
            sample_shape: Shape of samples to generate

        Returns:
            Extended shape combining sample_shape and batch_shape
        """
        return self.dist()._extended_shape(sample_shape)

    def _get_checked_instance(self, cls, _instance=None):
        """Get a checked instance for distribution operations."""
        return self.dist()._get_checked_instance(cls, _instance)


# KL Divergence Registration
# ==========================
# These functions register KL divergence computation methods for TensorDistribution
# with PyTorch's KL divergence registry, enabling seamless interoperability between
# TensorDistribution and standard torch.distributions.Distribution instances.


@register_kl(TensorDistribution, TensorDistribution)
def register_td_td(
    td_a: TensorDistribution,
    td_b: TensorDistribution,
):
    """
    Compute KL divergence between two TensorDistribution instances.

    Args:
        td_a: First TensorDistribution (P in KL(P||Q))
        td_b: Second TensorDistribution (Q in KL(P||Q))

    Returns:
        KL divergence KL(td_a || td_b) computed using underlying distributions
    """
    return kl_divergence(td_a.dist(), td_b.dist())


@register_kl(TensorDistribution, Distribution)
def register_td_d(td: TensorDistribution, d: Distribution):
    """
    Compute KL divergence from TensorDistribution to standard Distribution.

    Args:
        td: TensorDistribution instance (P in KL(P||Q))
        d: Standard torch.distributions.Distribution (Q in KL(P||Q))

    Returns:
        KL divergence KL(td || d)
    """
    return kl_divergence(td.dist(), d)


@register_kl(Distribution, TensorDistribution)
def register_d_td(
    d: Distribution,
    td: TensorDistribution,
):
    """
    Compute KL divergence from standard Distribution to TensorDistribution.

    Args:
        d: Standard torch.distributions.Distribution (P in KL(P||Q))
        td: TensorDistribution instance (Q in KL(P||Q))

    Returns:
        KL divergence KL(d || td)
    """
    return kl_divergence(d, td.dist())
