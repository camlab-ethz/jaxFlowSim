import numpy as np
from jaxtyping import Array, Float, Integer
from typing import TypeAlias

SimDat: TypeAlias = Float[Array, "5 K"]  # type: ignore
StaticSimDat: TypeAlias = np.ndarray
SimDatAux: TypeAlias = Float[Array, "N 3"]  # type: ignore
StaticSimDatAux: TypeAlias = np.ndarray
SimDatConst: TypeAlias = Float[Array, "7 K"]  # type: ignore
StaticSimDatConst: TypeAlias = np.ndarray
SimDatConstAux: TypeAlias = Float[Array, "N 7"]  # type: ignore
StaticSimDatConstAux: TypeAlias = np.ndarray
Timepoints: TypeAlias = Float[Array, " J"]  # type: ignore
StaticTimepoints: TypeAlias = np.ndarray
ScalarFloat: TypeAlias = Float[Array, ""]  # type: ignore
StaticScalarFloat: TypeAlias = float
StaticScalarInt: TypeAlias = int
InputData: TypeAlias = Float[Array, "2*N H"]  # type: ignore
StaticInputData: TypeAlias = np.ndarray
Masks: TypeAlias = Integer[Array, "2 K"]  # type: ignore
StaticMasks: TypeAlias = np.ndarray
Strides: TypeAlias = Integer[Array, "N 5"]  # type: ignore
StaticStrides: TypeAlias = np.ndarray
Edges: TypeAlias = Integer[Array, "N 10"]  # type: ignore
StaticEdges: TypeAlias = np.ndarray
TimepointsReturn: TypeAlias = Float[Array, " I"]  # type: ignore
StaticTimepointsReturn: TypeAlias = np.ndarray
PressureReturn: TypeAlias = Float[Array, "I 5*N"]  # type: ignore
StaticPressureReturn: TypeAlias = np.ndarray
String: TypeAlias = str
Strings: TypeAlias = list[str]
Bool: TypeAlias = bool
