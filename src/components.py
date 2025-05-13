"""
This module provides a class to represent the properties of blood in a vascular network model using JAX.

It includes a class to:
- Define the properties of blood (`Blood`).

The module makes use of the following imported utilities:
- `jaxtyping` and `beartype` for type checking and ensuring type safety in the functions.
"""

from jaxtyping import jaxtyped
from beartype import beartype as typechecker

from src.types import StaticScalarFloat


@jaxtyped(typechecker=typechecker)
class Blood:
    """
    A class to represent the properties of blood.

    Attributes:
    mu (Float[Array, ""]): Dynamic viscosity of blood.
    rho (Float[Array, ""]): Density of blood.
    rho_inv (Float[Array, ""]): Inverse density of blood.
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
        self.mu = mu
        self.rho = rho
        self.rho_inv = rho_inv
