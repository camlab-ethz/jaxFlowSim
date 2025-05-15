"""
This module provides functions to calculate pressure and wave speed in blood vessels.

It includes the following functions:
- `pressure`: Computes the pressure in a vessel given the cross-sectional area, reference area, stiffness coefficient, and external pressure.
- `pressure_sa`: Computes the pressure in a vessel given the square root of the area ratio, stiffness coefficient, and external pressure.
- `wave_speed`: Calculates the wave speed in a vessel given the cross-sectional area and admittance coefficient.
- `wave_speed_sa`: Calculates the wave speed in a vessel given the square root of the area and admittance coefficient.

The module makes use of the following imported utilities:
- `jax.numpy` for numerical operations and array handling.
- `jaxtyping` and `beartype` for type checking and ensuring type safety in the functions.
"""

import jax.numpy as jnp
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

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
    Computes the pressure in a vessel given the cross-sectional area, reference area, stiffness coefficient, and external pressure.

    Parameters:
    a (Float[Array, "..."]): Cross-sectional area of the vessel.
    a0 (Float[Array, "..."]): Reference cross-sectional area of the vessel.
    beta (Float[Array, "..."]): Stiffness coefficient of the vessel.
    p_ext (Float[Array, "..."]): External pressure.

    Returns:
    Float[Array, "..."]: Computed pressure in the vessel.
    """
    return p_ext + beta * (jnp.sqrt(a / a0) - 1.0)


@jaxtyped(typechecker=typechecker)
def pressure_sa(
    s_a_over_a0: ScalarFloat | SimDatSingle | StaticSimDatSingle,
    beta: ScalarFloat | SimDatSingle | StaticSimDatSingle,
    p_ext: ScalarFloat | StaticScalarFloat,
) -> ScalarFloat | SimDatSingle | StaticSimDatSingle:
    """
    Computes the pressure in a vessel given the square root of the area ratio, stiffness coefficient, and external pressure.

    Parameters:
    s_a_over_a0 (Float[Array, "..."]): Square root of the ratio of the cross-sectional area to the reference area.
    beta (Float[Array, "..."]): Stiffness coefficient of the vessel.
    p_ext (Float[Array, "..."]): External pressure.

    Returns:
    Float[Array, "..."]: Computed pressure in the vessel.
    """
    return p_ext + beta * (s_a_over_a0 - 1.0)


@jaxtyped(typechecker=typechecker)
def wave_speed(
    a: PairFloat | TripleFloat | StaticSimDatSingle,
    gamma: PairFloat | TripleFloat | StaticSimDatSingle,
) -> PairFloat | TripleFloat | SimDatSingle:
    """
    Calculates the wave speed in a vessel given the cross-sectional area and admittance coefficient.

    Parameters:
    a (Float[Array, "..."]): Cross-sectional area of the vessel.
    gamma (Float[Array, "..."]): Admittance coefficient of the vessel.

    Returns:
    Float[Array, "..."]: Computed wave speed in the vessel.
    """
    return jnp.sqrt(1.5 * gamma * jnp.sqrt(a))


@jaxtyped(typechecker=typechecker)
def wave_speed_sa(
    sa: ScalarFloat | StaticScalarFloat, gamma: ScalarFloat | StaticScalarFloat
) -> ScalarFloat:
    """
    Calculates the wave speed in a vessel given the square root of the area and admittance coefficient.

    Parameters:
    sa (Float[Array, "..."]): Square root of the cross-sectional area of the vessel.
    gamma (Float[Array, "..."]): Admittance coefficient of the vessel.

    Returns:
    Float[Array, "..."]: Computed wave speed in the vessel.
    """
    return jnp.sqrt(1.5 * gamma * sa)
