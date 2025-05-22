"""
utils.py

This module provides physical models for blood vessels, including pressure and wave speed
calculations based on vessel geometry and material properties.

Functions:
    pressure(a, a0, beta, p_ext)
        Calculate transmural pressure from cross-sectional areas and stiffness.
    pressure_sa(s_a_over_a0, beta, p_ext)
        Calculate transmural pressure using precomputed square-root area ratio.
    wave_speed(a, gamma)
        Compute wave speed from vessel area and admittance coefficient.
    wave_speed_sa(sa, gamma)
        Compute wave speed using square-root area directly.

All functions support JAX arrays and are decorated with type checking via jaxtyping and beartype.
"""

import jax.numpy as jnp
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

# Import custom type aliases for shape and static dimension enforcement
from src.types import (
    PairFloat,
    ScalarFloat,
    SimDatSingle,
    StaticScalarFloat,
    StaticSimDatSingle,
    TripleFloat,
)


@jaxtyped(typechecker=typechecker)
def pressure(
    a: ScalarFloat | SimDatSingle,
    a0: ScalarFloat | SimDatSingle,
    beta: ScalarFloat | SimDatSingle,
    p_ext: ScalarFloat | SimDatSingle,
) -> ScalarFloat | SimDatSingle:
    """
    Compute the transmural pressure in a blood vessel.

    The pressure is given by:
        p = p_ext + beta * (sqrt(a/a0) - 1)
    where:
        a     : current cross-sectional area
        a0    : reference (unstressed) cross-sectional area
        beta  : vessel stiffness coefficient
        p_ext : external (perivascular) pressure

    Supports broadcasting: a, a0, beta, p_ext can be scalars or arrays of matching shape.

    Parameters
    ----------
    a : ScalarFloat | SimDatSingle
        Current vessel cross-sectional area (e.g., cm^2).
    a0 : ScalarFloat | SimDatSingle
        Reference (unstressed) cross-sectional area.
    beta : ScalarFloat | SimDatSingle
        Stiffness coefficient (pressure units).
    p_ext : ScalarFloat | SimDatSingle
        External pressure acting on the vessel wall.

    Returns
    -------
    ScalarFloat | SimDatSingle
        Computed transmural pressure (same shape as inputs).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> pressure(jnp.array(1.2), jnp.array(1.0), jnp.array(300.0), jnp.array(5.0))
    DeviceArray(54.641..., dtype=float32)
    """
    # compute square-root of area ratio
    sratio = jnp.sqrt(a / a0)
    # calculate transmural pressure
    p = p_ext + beta * (sratio - 1.0)
    return p


@jaxtyped(typechecker=typechecker)
def pressure_sa(
    s_a_over_a0: ScalarFloat | SimDatSingle | StaticSimDatSingle,
    beta: ScalarFloat | SimDatSingle | StaticSimDatSingle,
    p_ext: ScalarFloat | StaticScalarFloat,
) -> ScalarFloat | SimDatSingle | StaticSimDatSingle:
    """
    Compute transmural pressure using precomputed square-root area ratio.

    This variant avoids redundant sqrt(a/a0) calculations when
    the square-root ratio is already available.

    p = p_ext + beta * (s_a_over_a0 - 1)

    Parameters
    ----------
    s_a_over_a0 : ScalarFloat | SimDatSingle | StaticSimDatSingle
        Precomputed sqrt(a/a0).
    beta : ScalarFloat | SimDatSingle | StaticSimDatSingle
        Stiffness coefficient.
    p_ext : ScalarFloat | StaticScalarFloat
        External pressure.

    Returns
    -------
    ScalarFloat | SimDatSingle | StaticSimDatSingle
        Transmural pressure.

    Notes
    -----
    The inputs must satisfy s_a_over_a0 = sqrt(a/a0).
    This function can slightly reduce computation time in tight loops.
    """
    # apply linear relationship around reference state
    return p_ext + beta * (s_a_over_a0 - 1.0)


@jaxtyped(typechecker=typechecker)
def wave_speed(
    a: PairFloat | TripleFloat | StaticSimDatSingle,
    gamma: PairFloat | TripleFloat | StaticSimDatSingle,
) -> PairFloat | TripleFloat | SimDatSingle:
    """
    Calculate the pulse wave speed in a vessel segment.

    Derived from the Moens-Korteweg equation simplified form:
        c = sqrt(1.5 * gamma * sqrt(a))
    where:
        a     : cross-sectional area
        gamma : admittance coefficient (area^-1 units)

    Parameters
    ----------
    a : PairFloat | TripleFloat | StaticSimDatSingle
        Cross-sectional area array; supports 2D or 3D vessel data.
    gamma : PairFloat | TripleFloat | StaticSimDatSingle
        Admittance coefficient matching shape of a.

    Returns
    -------
    PairFloat | TripleFloat | SimDatSingle
        Wave speed array in same shape as inputs.
    """
    # inner sqrt for a
    sqrt_a = jnp.sqrt(a)
    # apply coefficient and outer sqrt
    c = jnp.sqrt(1.5 * gamma * sqrt_a)
    return c


@jaxtyped(typechecker=typechecker)
def wave_speed_sa(
    sa: ScalarFloat | StaticScalarFloat,
    gamma: ScalarFloat | StaticScalarFloat,
) -> ScalarFloat:
    """
    Compute pulse wave speed from sqrt(area) directly.

    Offers slight performance gain if sa = sqrt(a) is known.

    c = sqrt(1.5 * gamma * sa)

    Parameters
    ----------
    sa : ScalarFloat | StaticScalarFloat
        Precomputed square-root of cross-sectional area.
    gamma : ScalarFloat | StaticScalarFloat
        Admittance coefficient.

    Returns
    -------
    ScalarFloat
        Scalar wave speed value.
    """
    return jnp.sqrt(1.5 * gamma * sa)
