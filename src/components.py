"""
Module for blood rheological properties in vascular network simulations using JAX.

This module defines the `Blood` class, which encapsulates key hemodynamic
parameters—dynamic viscosity, density, and its reciprocal—used throughout
the vascular network model.

Classes
-------
Blood
    Container for blood properties: viscosity (μ), density (ρ), and inverse density (1/ρ).

Dependencies
------------
- `jaxtyping.jaxtyped` : Static type annotations for JAX-compatible arrays.
- `beartype.beartype`  : Runtime type checking of function and class signatures.
- `src.types.StaticScalarFloat` : Alias for a scalar float type in the simulation.
- jax.numpy (jnp)   : Array operations and numerical routines.
"""

from jaxtyping import jaxtyped
from beartype import beartype as typechecker
import jax.numpy as jnp

from src.types import StaticScalarFloat


@jaxtyped(typechecker=typechecker)
class Blood:
    """
    Container for blood rheology parameters.

    Attributes
    ----------
    mu : StaticScalarFloat
        Dynamic (shear) viscosity of blood, in Pascal-seconds (Pa·s).
    rho : StaticScalarFloat
        Density of blood, in kilograms per cubic meter (kg/m³).
    rho_inv : StaticScalarFloat
        Reciprocal of blood density (1/ρ), in cubic meters per kilogram (m³/kg).

    Notes
    -----
    - `rho_inv` is provided as a convenience to avoid repeated divisions by `rho`
      during simulation loops.
    - All attributes are annotated for JAX compatibility and runtime type safety.
    """

    mu: StaticScalarFloat
    rho: StaticScalarFloat
    rho_inv: StaticScalarFloat

    def __init__(
        self,
        mu: StaticScalarFloat,
        rho: StaticScalarFloat,
        rho_inv: StaticScalarFloat,
    ):
        """
        Initialize a Blood instance with given physical properties.

        Parameters
        ----------
        mu : StaticScalarFloat
            Dynamic viscosity of blood (Pa·s).
        rho : StaticScalarFloat
            Density of blood (kg/m³).
        rho_inv : StaticScalarFloat
            Reciprocal of the density (m³/kg). Must equal 1.0 / rho.

        Raises
        ------
        ValueError
            If `rho_inv` does not match `1.0 / rho` within machine precision.
        """
        # Assign attributes
        self.mu = mu
        self.rho = rho
        self.rho_inv = rho_inv

        # Validate that rho_inv is consistent with rho
        expected_inv = 1.0 / rho
        if not jnp.isclose(rho_inv, expected_inv):
            raise ValueError(
                f"rho_inv ({rho_inv}) must equal 1.0 / rho ({expected_inv})"
            )
