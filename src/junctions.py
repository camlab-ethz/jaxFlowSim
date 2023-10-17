import jax
from src.conjunctions import solveConjunction


def joinVessels(*args):
    if len(args) == 6:
        return solveConjunction(args[0], args[1], args[2], args[3], args[4], args[5])
    #elif len(vessels) == 9:
    #    return solveBifurcation(vessels[0], vessels[1], vessels[2])

