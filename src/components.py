"""
This module provides a class to represent the properties of blood in a vascular network model using JAX.

It includes a class to:
- Define the properties of blood (`Blood`).

The module makes use of the following imported utilities:
- `jaxtyping` and `typeguard` for type checking and ensuring type safety in the functions.
"""

from jaxtyping import Array, Float, jaxtyped
from typeguard import typechecked as typechecker


@jaxtyped(typechecker=typechecker)
class Blood:
    """
    A class to represent the properties of blood.

    Attributes:
    mu (Float[Array, ""]): Dynamic viscosity of blood.
    rho (Float[Array, ""]): Density of blood.
    rho_inv (Float[Array, ""]): Inverse density of blood.
    """

    mu: Float[Array, ""]
    rho: Float[Array, ""]
    rho_inv: Float[Array, ""]

    def __init__(
        self,
        mu: Float[Array, ""],
        rho: Float[Array, ""],
        rho_inv: Float[Array, ""],
    ):
        self.mu = mu
        self.rho = rho
        self.rho_inv = rho_inv
