import jax
import jax.numpy as jnp
from src.conjunctions import solveConjunction
from src.bifurcations import solveBifurcation


@jax.jit
def joinVessels(*vessels):
    if len(vessels) == 2:
        return solveConjunction(vessels[0], vessels[1])
    elif len(vessels) == 3:
        return solveBifurcation(vessels[0], vessels[1], vessels[2])

