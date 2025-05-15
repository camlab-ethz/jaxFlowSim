import numpy as np
from jaxtyping import Array, Float, Integer, Bool
from typing import TypeAlias

# TODO: remove use of np.ndarray and use jax types instead or make them more concise by using NDarray and e.g. np.float64
SimDat: TypeAlias = Float[Array, "5 K"]  # type: ignore
StaticSimDat: TypeAlias = np.ndarray
SimDatSingle: TypeAlias = Float[Array, " K"]  # type: ignore
SimDatSingleReduced: TypeAlias = Float[Array, " K-1"]  # type: ignore
SimDatDouble: TypeAlias = Float[Array, "2 K"]  # type: ignore
StaticSimDatSingle: TypeAlias = np.ndarray
SimDatAux: TypeAlias = Float[Array, "N 3"]  # type: ignore
SimDatAuxSingle: TypeAlias = Float[Array, " N"]  # type: ignore
StaticSimDatAux: TypeAlias = np.ndarray
SimDatConst: TypeAlias = Float[Array, "7 K"]  # type: ignore
StaticSimDatConst: TypeAlias = np.ndarray
SimDatConstAux: TypeAlias = Float[Array, "N 7"]  # type: ignore
StaticSimDatConstAux: TypeAlias = np.ndarray
Timepoints: TypeAlias = Float[Array, " J"]  # type: ignore
StaticTimepoints: TypeAlias = np.ndarray
ScalarFloat: TypeAlias = Float[Array, ""]
PairFloat: TypeAlias = Float[Array, "2"]
TripleFloat: TypeAlias = Float[Array, "3"]
QuadFloat: TypeAlias = Float[Array, "4"]
HexaFloat: TypeAlias = Float[Array, "6"]
SmallJacobian: TypeAlias = Float[Array, "4 4"]
LargeJacobian: TypeAlias = Float[Array, "6 6"]
DoubleJunctionReturn: TypeAlias = Float[Array, "10"]
TripleJunctionReturn: TypeAlias = Float[Array, "15"]
StaticScalarFloat: TypeAlias = float | np.floating
StaticScalarInt: TypeAlias = int | np.int64
ScalarInt: TypeAlias = Integer[Array, ""]
# TODO: implicitly implement S=2*N
InputData: TypeAlias = Float[Array, "S H"]  # type: ignore
StaticInputData: TypeAlias = np.ndarray
StaticInputDataSingle: TypeAlias = np.ndarray
Masks: TypeAlias = Integer[Array, "2 K"]  # type: ignore
MasksPadded: TypeAlias = Integer[Array, "2 K+4"]  # type: ignore
StaticMasks: TypeAlias = np.ndarray
Strides: TypeAlias = Integer[Array, "N 5"]  # type: ignore
StridesReduced: TypeAlias = Integer[Array, "N 2"]  # type: ignore
StaticStrides: TypeAlias = np.ndarray
Edges: TypeAlias = Integer[Array, "N 10"]  # type: ignore
StaticEdges: TypeAlias = np.ndarray
TimepointsReturn: TypeAlias = Float[Array, " I"]
StaticTimepointsReturn: TypeAlias = np.ndarray
# TODO: implicitly implement M=5*N
PressureReturn: TypeAlias = Float[Array, "I M"]
PressureReturnSingle: TypeAlias = Float[Array, " I"]
StaticPressureReturn: TypeAlias = np.ndarray
String: TypeAlias = str
Strings: TypeAlias = list[str]
StaticBool: TypeAlias = bool
ScalarBool: TypeAlias = Bool[Array, ""]
Dict: TypeAlias = dict
Dicts: TypeAlias = list[dict]

SimulationStepArgs = tuple[
    SimDat,
    SimDatAux,
    SimDatConst,
    SimDatConstAux,
    ScalarFloat,
    ScalarInt,
    Timepoints,
    ScalarInt,
    ScalarFloat,
    PressureReturn,
    PressureReturn,
    TimepointsReturn,
    ScalarFloat,
    ScalarFloat,
    Edges,
    InputData,
    ScalarFloat,
]

SimulationStepArgsUnsafe: TypeAlias = tuple[
    SimDat,
    SimDatAux,
    SimDatConst,
    SimDatConstAux,
    ScalarFloat,
    ScalarFloat,
    TimepointsReturn,
    Edges,
    InputData,
    ScalarFloat,
    PressureReturn,
]
