import inspect
from typing import Type

import pytest
import torch
from torch.distributions import Distribution

from tensorcontainer.tensor_distribution import TensorDistribution

compile_args = [
    {"fullgraph": True, "dynamic": True},
    {"fullgraph": True, "dynamic": False},
    {"fullgraph": False, "dynamic": True},
    {"fullgraph": False, "dynamic": False},
]


def assert_tensor_dist_api_matches_torch_dist_api(
    tdist: TensorDistribution, torch_dist: Distribution
) -> None:
    """
    Asserts that the API of a TensorDistribution instance matches its corresponding
    torch.distributions.Distribution instance.

    This function compares various methods and properties (sample, log_prob, mean, variance, etc.)
    between the two distribution instances to ensure consistency.

    Args:
        tdist (TensorDistribution): The TensorDistribution instance.
        torch_dist (Distribution): The corresponding torch.distributions.Distribution instance.
    """
    # Test sample method
    sample_shape = torch.Size([2, 3])
    td_sample = tdist.sample(sample_shape)
    torch_sample = torch_dist.sample(sample_shape)
    assert td_sample.shape == torch_sample.shape
    assert td_sample.dtype == torch_sample.dtype
    assert td_sample.device == torch_sample.device

    # Test log_prob method
    # For Wishart, the sample is a matrix, so we need to ensure the sample has the correct event shape
    if tdist.event_shape:  # Check if event_shape is not empty
        # Create a dummy value with the correct event shape and batch shape for log_prob
        # This might need to be more sophisticated for complex distributions
        value_shape = sample_shape + tdist.event_shape
        value = torch.randn(value_shape, dtype=tdist.dtype, device=tdist.device)
    else:
        value = torch.randn(sample_shape, dtype=tdist.dtype, device=tdist.device)

    td_log_prob = tdist.log_prob(value)
    torch_log_prob = torch_dist.log_prob(value)
    assert td_log_prob.shape == torch_log_prob.shape
    assert td_log_prob.dtype == torch_log_prob.dtype
    assert td_log_prob.device == torch_log_prob.device
    # Check if values are close, as log_prob can have numerical differences
    torch.testing.assert_close(td_log_prob, torch_log_prob, rtol=1e-4, atol=1e-4)

    # Test mean property
    td_mean = tdist.mean
    torch_mean = torch_dist.mean
    assert td_mean.shape == torch_mean.shape
    assert td_mean.dtype == torch_mean.dtype
    assert td_mean.device == torch_mean.device
    torch.testing.assert_close(td_mean, torch_mean, rtol=1e-4, atol=1e-4)

    # Test variance property
    td_variance = tdist.variance
    torch_variance = torch_dist.variance
    assert td_variance.shape == torch_variance.shape
    assert td_variance.dtype == torch_variance.dtype
    assert td_variance.device == torch_variance.device
    torch.testing.assert_close(td_variance, torch_variance, rtol=1e-4, atol=1e-4)

    # Test batch_shape and event_shape properties
    assert tdist.batch_shape == torch_dist.batch_shape
    assert tdist.event_shape == torch_dist.event_shape

    # Test has_rsample property
    assert tdist.has_rsample == torch_dist.has_rsample

    # Test entropy method (if supported)
    if hasattr(torch_dist, "entropy"):
        td_entropy = tdist.entropy()
        torch_entropy = torch_dist.entropy()
        assert td_entropy.shape == torch_entropy.shape
        assert td_entropy.dtype == torch_entropy.dtype
        assert td_entropy.device == torch_entropy.device
        torch.testing.assert_close(td_entropy, torch_entropy, rtol=1e-4, atol=1e-4)

    # Test cdf method (if supported)
    if hasattr(torch_dist, "cdf"):
        td_cdf = tdist.cdf(value)
        torch_cdf = torch_dist.cdf(value)
        assert td_cdf.shape == torch_cdf.shape
        assert td_cdf.dtype == torch_cdf.dtype
        assert td_cdf.device == torch_cdf.device
        torch.testing.assert_close(td_cdf, torch_cdf, rtol=1e-4, atol=1e-4)

    # Test icdf method (if supported)
    if hasattr(torch_dist, "icdf"):
        # Create a dummy probability value for icdf
        prob_shape = sample_shape + tdist.event_shape
        prob = torch.rand(prob_shape, dtype=tdist.dtype, device=tdist.device)
        td_icdf = tdist.icdf(prob)
        torch_icdf = torch_dist.icdf(prob)
        assert td_icdf.shape == torch_icdf.shape
        assert td_icdf.dtype == torch_icdf.dtype
        assert td_icdf.device == torch_icdf.device
        torch.testing.assert_close(td_icdf, torch_icdf, rtol=1e-4, atol=1e-4)


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(0)


@pytest.fixture(autouse=True)
def preserve_distributions_validation():
    """
    Preserve and restore torch.distributions validation state around each test.
    This prevents torch.compile usage in one test from affecting subsequent tests.

    This fixture automatically runs before and after every test in this directory,
    ensuring that any global state changes (like torch.compile disabling validation)
    don't leak between tests.
    """
    # Store the original validation state before the test
    original_validate_args = torch.distributions.Distribution._validate_args

    yield  # Run the test

    # Restore the original validation state after the test
    torch.distributions.Distribution.set_default_validate_args(original_validate_args)


@pytest.fixture(autouse=True)
def with_distributions_validation():
    """
    Fixture to ensure distributions validation is enabled for specific tests.

    Usage:
        def test_validation_required(self, with_distributions_validation):
            # Validation is guaranteed to be enabled in this test
            with pytest.raises(ValueError):
                TensorNormal(loc=torch.tensor([1.0]), scale=torch.tensor([-0.1]))
    """
    # Force enable validation for this test
    torch.distributions.Distribution.set_default_validate_args(True)
    yield


def normalize_device(dev: torch.device) -> torch.device:
    d = torch.device(dev)
    # If no index was given, fill in current_device() for CUDA, leave CPU as-is
    if d.type == "cuda" and d.index is None:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()  # e.g. 0
            return torch.device(f"cuda:{idx}")
        else:
            # If CUDA is not available, return the device as-is
            return d
    return d


def assert_init_signatures_match(
    td_class: Type[TensorDistribution], torch_dist_class: Type[Distribution]
) -> None:
    """
    Assert that __init__ signatures match between TensorDistribution and Distribution.

    This function compares the __init__ signatures of a TensorDistribution subclass
    and a torch.distributions.Distribution subclass, asserting that they are
    identical, disregarding the 'self' and 'validate_args' parameters.

    Args:
        td_class (Type[TensorDistribution]): The TensorDistribution class.
        torch_dist_class (Type[Distribution]): The torch.distribution.Distribution class.

    """
    td_sig = inspect.signature(td_class.__init__)
    torch_sig = inspect.signature(torch_dist_class.__init__)

    td_params = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in td_sig.parameters.values()
        if p.name not in ("self", "validate_args")
    ]
    torch_params = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in torch_sig.parameters.values()
        if p.name not in ("self", "validate_args", "eps")
    ]

    td_sig_compare = td_sig.replace(parameters=td_params)
    torch_sig_compare = torch_sig.replace(parameters=torch_params)

    # Compare string representations of return annotations to handle __future__.annotations
    td_return_annotation_str = str(td_sig_compare.return_annotation)
    torch_return_annotation_str = str(torch_sig_compare.return_annotation)

    # Create new signatures with empty return annotations for comparison
    td_sig_no_return = td_sig_compare.replace(return_annotation=inspect.Signature.empty)
    torch_sig_no_return = torch_sig_compare.replace(
        return_annotation=inspect.Signature.empty
    )

    assert (
        td_sig_no_return == torch_sig_no_return
        and td_return_annotation_str == torch_return_annotation_str
    ), (
        f"__init__ signatures do not match between {td_class.__name__} and "
        f"{torch_dist_class.__name__}.\n"
        f"Got:      {td_class.__name__}{td_sig_compare}\n"
        f"Expected: {torch_dist_class.__name__}{torch_sig_compare}"
    )


def _get_torch_dist_properties(
    torch_dist_class: Type[Distribution],
) -> list[tuple[str, property]]:
    """
    Discover relevant public properties from a torch.distribution.Distribution class.
    This function inspects a torch.distribution.Distribution subclass and returns a
    list of its public properties, filtering out those that are defined in the base
    Distribution class to focus only on subclass-specific properties.
    Args:
        torch_dist_class (Type[Distribution]): The torch distribution class.
    Returns:
        A list of tuples, where each tuple contains the name and property object.
    """
    # Get all properties from the subclass
    subclass_properties = inspect.getmembers(
        torch_dist_class, predicate=lambda x: isinstance(x, property)
    )

    # Get all properties from the base Distribution class
    base_properties = inspect.getmembers(
        Distribution, predicate=lambda x: isinstance(x, property)
    )

    # Create a set of base class property names for efficient lookup
    base_property_names = {name for name, _ in base_properties}

    properties = []
    # Iterate over all public properties found in the torch.distribution subclass.
    for name, prop in subclass_properties:
        # Skip private properties, following the convention of an initial underscore.
        if name.startswith("_"):
            continue

        # Skip properties that are defined in the base Distribution class
        if name in base_property_names:
            continue

        properties.append((name, prop))
    return properties


def assert_properties_signatures_match(
    td_class: Type[TensorDistribution], torch_dist_class: Type[Distribution]
) -> None:
    """
    Asserts that public properties of a torch.distribution.Distribution are mirrored
    in the corresponding TensorDistribution subclass.

    This function verifies that for each public property in the torch distribution,
    there is a corresponding property in the TensorDistribution with a matching
    getter signature. This ensures that the TensorDistribution subclass maintains a
    consistent interface with the original torch distribution.

    Args:
        td_class (Type[TensorDistribution]): The TensorDistribution class to check.
        torch_dist_class (Type[Distribution]): The corresponding torch distribution class.
    """
    dist_properties = _get_torch_dist_properties(torch_dist_class)

    # Iterate over all public properties found in the torch.distribution class.
    for name, prop in dist_properties:
        # Check if the TensorDistribution class has the same property.
        assert hasattr(td_class, name), (
            f"{td_class.__name__} is missing property '{name}' from {torch_dist_class.__name__}"
        )
        td_prop = getattr(td_class, name)
        assert isinstance(td_prop, property), (
            f"Attribute '{name}' in {td_class.__name__} is not a property"
        )

        # If the torch distribution property has no getter, skip it.
        if prop.fget is None:
            continue

        # Access the getter method (fget) for both the TensorDistribution and torch.distribution properties.
        assert td_prop.fget is not None, (
            f"Property '{name}' in {td_class.__name__} is missing a getter"
        )

        # Retrieve the signatures of the getter methods using inspect.signature.
        td_getter_sig = inspect.signature(td_prop.fget)
        dist_getter_sig = inspect.signature(prop.fget)

        # Remove annotations from parameters for a less strict comparison, focusing on names and kinds.
        td_getter_params = [
            p.replace(annotation=inspect.Parameter.empty)
            for p in td_getter_sig.parameters.values()
        ]
        dist_getter_params = [
            p.replace(annotation=inspect.Parameter.empty)
            for p in dist_getter_sig.parameters.values()
        ]

        # Create comparable versions of the signatures, ignoring return annotations.
        td_getter_sig_compare = td_getter_sig.replace(
            parameters=td_getter_params,
            return_annotation=inspect.Signature.empty,
        )
        dist_getter_sig_compare = dist_getter_sig.replace(
            parameters=dist_getter_params,
            return_annotation=inspect.Signature.empty,
        )

        # Assert that the signatures of the getter methods match.
        assert td_getter_sig_compare == dist_getter_sig_compare, (
            f"Getter signature mismatch for property '{name}':\n"
            f"  {td_class.__name__}: {td_getter_sig}\n"
            f"  {torch_dist_class.__name__}: {dist_getter_sig}"
        )


def assert_property_values_match(td: TensorDistribution) -> None:
    """
    Asserts that property values match between a TensorDistribution and the
    torch.distribution.Distribution instance it wraps.

    This function iterates through the public properties of the underlying torch
    distribution, retrieves the value of each property from both the `td` instance
    and its corresponding `td.dist()` instance, and asserts that the values are
    identical. For tensor values, `torch.testing.assert_close` is used for robust
    comparison.

    Args:
        td (TensorDistribution): The TensorDistribution instance to check.
    """
    dist = td.dist()
    properties_to_check = _get_torch_dist_properties(dist.__class__)

    for name, _ in properties_to_check:
        td_value = getattr(td, name)
        dist_value = getattr(dist, name)

        if isinstance(td_value, torch.Tensor):
            try:
                torch.testing.assert_close(td_value, dist_value)
            except AssertionError as e:
                raise AssertionError(
                    f"Tensor value mismatch for property '{name}': {str(e)}"
                ) from e
        else:
            if isinstance(td_value, TensorDistribution):
                # Compare parameters of the underlying torch.Distribution instances
                td_dist = td_value.dist()
                dist_dist = dist_value

                # Check if they are the same class
                assert td_dist.__class__ == dist_dist.__class__, (
                    f"Class mismatch for property '{name}':\n"
                    f"  {td.__class__.__name__}: {td_dist.__class__.__name__}\n"
                    f"  {dist.__class__.__name__}: {dist_dist.__class__.__name__}"
                )

                # Compare parameters (e.g., loc, scale, probs, logits)
                # This assumes that the parameters are accessible as attributes
                # and are either Tensors or simple Python types.
                # This might need to be more robust for complex distributions.
                for param_name in td_dist.arg_constraints.keys():
                    td_param = getattr(td_dist, param_name)
                    dist_param = getattr(dist_dist, param_name)

                    if isinstance(td_param, torch.Tensor):
                        torch.testing.assert_close(
                            td_param,
                            dist_param,
                            msg=f"Parameter '{param_name}' mismatch for property '{name}'",
                        )
                    else:
                        assert td_param == dist_param, (
                            f"Parameter '{param_name}' mismatch for property '{name}'"
                        )
            else:
                # For other non-tensor values, use direct equality check
                assert td_value == dist_value, (
                    f"Value mismatch for property '{name}':\n"
                    f"  {td.__class__.__name__}: {td_value}\n"
                    f"  {dist.__class__.__name__}: {dist_value}"
                )
