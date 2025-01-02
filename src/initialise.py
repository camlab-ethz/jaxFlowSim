"""
This module provides functions to handle configuration, build blood properties, and construct an arterial network for a vascular model using JAX.

It includes functions to:
- Load and validate configuration files (`load_config`, `load_yaml_file`, `check_input_file`, `check_sections`, `check_section`, `check_network`, `check_vessel`).
- Manage results folders (`make_results_folder`, `copy_inlet_files`).
- Build blood properties and arterial networks (`build_blood`, `build_arterial_network`, `build_vessel`).
- Compute various vessel-related parameters (`compute_radius_slope`, `compute_thickness`, `compute_radii`, `get_pext`, `get_phi`, `mesh_vessel`, `initialise_thickness`, `add_outlet`, `compute_viscous_term`, `build_heart`, `compute_windkessel_inlet_impedance`).

The module makes use of the following imported utilities:
- `Blood` from `src.components` for representing blood properties.
- `pressureSA` and `waveSpeed` from `src.utils` for calculating pressure and wave speed in the vessels.
- `numpy` for numerical operations and array handling.
- `jaxtyping` and `typeguard` for type checking and ensuring type safety in the functions.
"""

import os.path
import shutil
from typing import Any

import numpy as np
import yaml
from jaxtyping import Array, jaxtyped
from numpy.typing import NDArray
from typeguard import typechecked as typechecker

from src.components import Blood
from src.utils import pressure_sa, wave_speed


@jaxtyped(typechecker=typechecker)
def load_config(input_filename: str) -> dict:
    """
    Loads the configuration from a YAML file and checks its validity.

    Parameters:
    input_filename (str): Path to the YAML configuration file.

    Returns:
    dict: Loaded configuration data.
    """
    data = load_yaml_file(input_filename)
    check_input_file(data)
    return data


@jaxtyped(typechecker=typechecker)
def load_yaml_file(input_filename: str) -> dict:
    """
    Loads a YAML file.

    Parameters:
    input_filename (str): Path to the YAML file.

    Returns:
    dict: Loaded YAML data.

    Raises:
    ValueError: If the file does not exist.
    """
    if not os.path.isfile(input_filename):
        raise ValueError(f"missing file {input_filename}")

    with open(input_filename, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@jaxtyped(typechecker=typechecker)
def check_input_file(data: dict) -> None:
    """
    Checks the validity of the input configuration file.

    Parameters:
    data (dict): Configuration data.

    Returns:
    None
    """
    check_sections(data)
    check_network(data["network"])


@jaxtyped(typechecker=typechecker)
def check_sections(data: dict) -> None:
    """
    Checks the presence of required sections in the configuration data.

    Parameters:
    data (dict): Configuration data.

    Returns:
    None

    Raises:
    ValueError: If any required section is missing.
    """
    keys = ["proj_name", "network", "blood", "solver"]
    for key in keys:
        if key not in data:
            raise ValueError(f"missing section {key} in YAML input file")

    check_section(data, "blood", ["mu", "rho"])
    check_section(data, "solver", ["Ccfl", "conv_tol"])

    if "num_snapshots" not in data["solver"]:
        data["solver"]["num_snapshots"] = 100


@jaxtyped(typechecker=typechecker)
def check_section(data: dict, section: str, keys: list[str]) -> None:
    """
    Checks the presence of required keys in a specific section of the configuration data.

    Parameters:
    data (dict): Configuration data.
    section (str): Section to check.
    keys (list[str]): Required keys in the section.

    Returns:
    None

    Raises:
    ValueError: If any required key is missing in the section.
    """
    for key in keys:
        if key not in data[section]:
            raise ValueError(f"missing {key} in {section} section")


@jaxtyped(typechecker=typechecker)
def check_network(network: list[dict]) -> None:
    """
    Checks the validity of the network configuration.

    Parameters:
    network (list[dict]): List of vessel configurations.

    Returns:
    None

    Raises:
    ValueError: If there are any issues with the network configuration.
    """
    has_inlet = False
    inlets = set()
    has_outlet = False
    nodes = {}

    for i, vessel in enumerate(network, start=1):
        check_vessel(i, vessel)

        if "inlet" in vessel:
            has_inlet = True
            inlet_node = vessel["sn"]
            if inlet_node in inlets:
                raise ValueError(f"inlet {inlet_node} used multiple times")
            inlets.add(vessel["sn"])
        if "outlet" in vessel:
            has_outlet = True

        # check max number of vessels per node
        if vessel["sn"] not in nodes:
            nodes[vessel["sn"]] = 1
        else:
            nodes[vessel["sn"]] += 1
        if vessel["tn"] not in nodes:
            nodes[vessel["tn"]] = 1
        else:
            nodes[vessel["tn"]] += 1
        if nodes[vessel["sn"]] > 3:
            raise ValueError(f"too many vessels connected at node {vessel['sn']}")
        elif nodes[vessel["tn"]] > 3:
            raise ValueError(f"too many vessels connected at node {vessel['tn']}")

    # outlet nodes must be defined
    for i, vessel in enumerate(network, start=1):
        if nodes[vessel["tn"]] == 1:
            if "outlet" not in vessel:
                raise ValueError(
                    f"outlet not defined for vessel {vessel['label']}, check connectivity"
                )

    if not has_inlet:
        raise ValueError("missing inlet(s) definition")

    if not has_outlet:
        raise ValueError("missing outlet(s) definition")


@jaxtyped(typechecker=typechecker)
def check_vessel(i: int, vessel: dict) -> None:
    """
    Checks the validity of a single vessel configuration.

    Parameters:
    i (int): Index of the vessel.
    vessel (dict): Vessel configuration data.

    Returns:
    None

    Raises:
    ValueError: If there are any issues with the vessel configuration.
    """
    keys = ["label", "sn", "tn", "L", "E"]
    for key in keys:
        if key not in vessel:
            raise ValueError(f"vessel {i} is missing {key} value")

    if vessel["sn"] == vessel["tn"]:
        raise ValueError(f"vessel {i} has same sn and tn")

    if "R0" not in vessel:
        if "Rp" not in vessel and "Rd" not in vessel:
            raise ValueError(f"vessel {i} is missing lumen radius value(s)")
    else:
        if vessel["R0"] > 0.05:
            print(f"{vessel['label']} radius larger than 5cm!")

    if "inlet" in vessel:
        if "inlet file" not in vessel:
            raise ValueError(f"inlet vessel {i} is missing the inlet file path")
        elif not os.path.isfile(vessel["inlet file"]):
            file_path = vessel["inlet file"]
            raise ValueError(f"vessel {i} inlet file {file_path} not found")

        if "inlet number" not in vessel:
            raise ValueError(f"inlet vessel {i} is missing the inlet number")

    if "outlet" in vessel:
        outlet = vessel["outlet"]
        if outlet == "wk3":
            if "R1" not in vessel or "Cc" not in vessel:
                raise ValueError(
                    f"outlet vessel {i} is missing three-element windkessel values"
                )
        elif outlet == "wk2":
            if "R1" not in vessel or "Cc" not in vessel:
                raise ValueError(
                    f"outlet vessel {i} is missing two-element windkessel values"
                )
        elif outlet == "reflection":
            if "Rt" not in vessel:
                raise ValueError(
                    f"outlet vessel {i} is missing reflection coefficient value"
                )


@jaxtyped(typechecker=typechecker)
def make_results_folder(data: dict, input_filename: str) -> None:
    """
    Creates the results folder for the simulation and copies necessary files.

    Parameters:
    data (dict): Configuration data.
    input_filename (str): Path to the input file.

    Returns:
    None
    """
    project_name = data["proj_name"]

    if "results folder" not in data:
        r_folder = "results/" + project_name + "_results"
    else:
        r_folder = data["results folder"]

    # delete existing folder and results
    if os.path.isdir(r_folder):
        shutil.rmtree(r_folder)

    os.makedirs(r_folder, mode=0o777)
    shutil.copy2(input_filename, r_folder + "/")
    copy_inlet_files(data, r_folder)


@jaxtyped(typechecker=typechecker)
def copy_inlet_files(data: dict, r_folder: str) -> None:
    """
    Copies inlet files to the results folder.

    Parameters:
    data (dict): Configuration data.
    r_folder (str): Path to the results folder.

    Returns:
    None
    """
    for vessel in data["network"]:
        if "inlet file" in vessel:
            shutil.copy2(vessel["inlet file"], r_folder + "/")


@jaxtyped(typechecker=typechecker)
def build_blood(blood_data: dict) -> Blood:
    """
    Builds the Blood object from the provided data.

    Parameters:
    blood_data (dict): Data containing blood properties.

    Returns:
    Blood: Blood object with specified properties.
    """
    mu = blood_data["mu"]
    rho = blood_data["rho"]
    rho_inv = 1.0 / rho

    return Blood(mu, rho, rho_inv)


@jaxtyped(typechecker=typechecker)
def build_arterial_network(network: list[dict], blood: Blood) -> tuple[
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    int,
    int,
    NDArray,
    NDArray,
    NDArray,
    list[str],
    NDArray,
]:
    """
    Builds the arterial network from the provided configuration data.

    Parameters:
    network (list[dict]): List of vessel configurations.
    blood (Blood): Blood object with specified properties.

    Returns:
    tuple: Contains arrays of simulation data, auxiliary data, constants, edges, vessel names, and input data.
    """

    b = 2
    n = len(network)
    m_0 = mesh_vessel(network[0], float(network[0]["L"]))
    starts = np.zeros(n, dtype=np.int64)
    ends = np.zeros(n, dtype=np.int64)

    starts[0] = b
    ends[0] = m_0 + b

    for i in range(1, n):
        length = float(network[i]["L"])
        _m = mesh_vessel(network[i], length)
        starts[i] = ends[i - 1] + 2 * b
        ends[i] = starts[i] + _m

    strides = np.zeros((n, 5), dtype=np.int64)
    strides[:, 0] = starts
    strides[:, 1] = ends

    k = ends[-1] + b
    starts_rep = np.zeros(k, dtype=np.int64)
    ends_rep = np.zeros(k, dtype=np.int64)

    for i in range(0, n):
        starts_rep[strides[i, 0] - b : strides[i, 1] + b] = strides[i, 0] * np.ones(
            strides[i, 1] - strides[i, 0] + 2 * b, np.int64
        )
        ends_rep[strides[i, 0] - b : strides[i, 1] + b] = strides[i, 1] * np.ones(
            strides[i, 1] - strides[i, 0] + 2 * b, np.int64
        )

    sim_dat = np.zeros((5, k), dtype=np.float64)
    sim_dat_aux = np.zeros((n, 3), dtype=np.float64)
    sim_dat_const = np.zeros((7, k), dtype=np.float64)
    sim_dat_const_aux = np.zeros((n, 7), dtype=np.float64)
    edges = np.zeros((n, 10), dtype=np.int64)
    input_data_temp = []
    vessel_names = []

    nodes = np.zeros((n, 3), dtype=np.int64)

    for i, vessel in enumerate(network):
        end = strides[i, 1]
        start = strides[i, 0]
        m = end - start

        (
            _edges,
            _input_data,
            _sim_dat,
            _sim_dat_aux,
            vessel_name,
            _sim_dat_const,
            _sim_dat_const_aux,
        ) = build_vessel(i + 1, vessel, blood, m)

        nodes[i, :] = (
            int(np.floor(m * 0.25)) - 1,
            int(np.floor(m * 0.5)) - 1,
            int(np.floor(m * 0.75)) - 1,
        )

        sim_dat[:, start:end] = _sim_dat
        sim_dat[:, start - b : start :] = (
            _sim_dat[:, 0, np.newaxis] * np.ones(b)[np.newaxis, :]
        )
        sim_dat[:, end : end + b] = (
            _sim_dat[:, -1, np.newaxis] * np.ones(b)[np.newaxis, :]
        )
        sim_dat_aux[i, 0:2] = _sim_dat_aux
        sim_dat_const[:, start:end] = _sim_dat_const
        sim_dat_const[:, start - b : start :] = (
            _sim_dat_const[:, 0, np.newaxis] * np.ones(b)[np.newaxis, :]
        )
        sim_dat_const[:, end : end + b] = (
            _sim_dat_const[:, -1, np.newaxis] * np.ones(b)[np.newaxis, :]
        )
        sim_dat_const_aux[i, :] = _sim_dat_const_aux

        edges[i, :3] = _edges
        input_data_temp.append(_input_data.transpose())

        sim_dat_const[-1, start - b : end + b] = sim_dat_const[
            -1, start - b : end + b
        ] / (m)
        vessel_names.append(vessel_name)

    input_sizes = [inpd.shape[1] for inpd in input_data_temp]
    input_size = max(input_sizes)
    input_data = np.ones((2 * n, input_size), dtype=np.float64) * 1000
    for i, inpd in enumerate(input_data_temp):
        input_data[2 * i : 2 * i + 2, : inpd.shape[1]] = inpd

    indices = np.arange(0, k, 1)
    indices_1 = indices - starts_rep == -starts_rep[0] + 1
    indices_2 = indices - ends_rep == -starts_rep[0] + 2
    masks = np.zeros((2, k), dtype=np.int64)
    masks[0, :] = indices_1
    masks[1, :] = indices_2

    for j in np.arange(0, edges.shape[0], 1):
        i = edges[j, 0] - 1
        if sim_dat_const_aux[i, 2] == 0:  # "none":
            t = edges[j, 2]
            edges[j, 3] = (
                np.where(
                    edges[:, 1] == t,
                    np.ones_like(edges[:, 1]),
                    np.zeros_like(edges[:, 1]),
                )
                .sum()
                .astype(int)
            )
            edges[j, 6] = (
                np.where(
                    edges[:, 2] == t,
                    np.ones_like(edges[:, 2]),
                    np.zeros_like(edges[:, 2]),
                )
                .sum()
                .astype(int)
            )
            if edges[j, 3] == 2:
                edges[j, 4] = np.where(edges[:, 1] == t)[0][0]
                edges[j, 5] = np.where(edges[:, 1] == t)[0][1]

            elif edges[j, 6] == 1:
                edges[j, 7] = np.where(edges[:, 1] == t)[0][0]

            elif edges[j, 6] == 2:
                temp_1 = np.where(edges[:, 2] == t)[0][0]
                temp_2 = np.where(edges[:, 2] == t)[0][1]
                edges[j, 7] = np.minimum(temp_1, temp_2)
                edges[j, 8] = np.maximum(temp_1, temp_2)
                edges[j, 9] = np.where(edges[:, 1] == t)[0][0]

    strides[:, 2:] = nodes

    return (
        sim_dat,
        sim_dat_aux,
        sim_dat_const,
        sim_dat_const_aux,
        n,
        b,
        masks,
        strides,
        edges,
        vessel_names,
        input_data,
    )


@jaxtyped(typechecker=typechecker)
def build_vessel(
    index: int, vessel_data: dict, blood: Blood, m: np.int64
) -> tuple[NDArray, NDArray, NDArray, NDArray, str, NDArray, NDArray]:
    """
    Builds the data for a single vessel.

    Parameters:
    index (int): Index of the vessel.
    vessel_data (dict): Configuration data for the vessel.
    blood (Blood): Blood object with specified properties.
    m (np.int64): Number of mesh points.

    Returns:
    tuple: Contains arrays of edges, input data, simulation data, auxiliary data, vessel name, constants, and auxiliary constants.
    """
    vessel_name = vessel_data["label"]
    s_n = int(vessel_data["sn"])
    t_n = int(vessel_data["tn"])
    length = float(vessel_data["L"])

    r_p, r_d = compute_radii(vessel_data)
    p_ext = get_pext(vessel_data)
    dx = length / m
    h_0 = initialise_thickness(vessel_data)
    outlet, rt, r1, r2, cc = add_outlet(vessel_data)
    visc_t = compute_viscous_term(vessel_data, blood)
    inlet, cardiac_t, input_data = build_heart(vessel_data)

    q = np.zeros(m, dtype=np.float64)
    u = np.zeros(m, dtype=np.float64)

    s_pi = np.sqrt(np.pi)
    one_over_rho_s_p = 1.0 / (3.0 * blood.rho * s_pi)
    radius_slope = compute_radius_slope(r_p, r_d, length)

    if h_0 == 0.0:
        r_mean = 0.5 * (r_p + r_d)
        h_0 = compute_thickness(r_mean)

    r_0 = radius_slope * np.arange(0, m, 1) * dx + r_p
    a_0 = np.pi * r_0 * r_0
    a = a_0
    beta = compute_beta(a_0, h_0, length, vessel_data)
    gamma = beta * one_over_rho_s_p / r_0
    c = wave_speed(a, gamma)
    wall_e = 3.0 * beta * radius_slope * 1 / a_0 * s_pi * blood.rho_inv
    p = pressure_sa(np.ones(m, np.float64), beta, p_ext)

    if outlet == "wk2":
        r1, r2 = compute_windkessel_inlet_impedance(r2, blood, a_0, gamma)
        outlet = "wk3"

    w1m0 = u[-1] - 4.0 * c[-1]
    w2m0 = u[-1] + 4.0 * c[-1]

    sim_dat = np.stack((u, q, a, c, p))
    sim_dat_aux = np.array([w1m0, w2m0])
    sim_dat_const = np.stack(
        (
            a_0,
            beta,
            gamma,
            wall_e,
            p_ext * np.ones(m),
            visc_t * np.ones(m),
            length * np.ones(m),
        )
    )
    sim_dat_const_aux = np.array([cardiac_t, inlet, outlet, rt, r1, r2, cc])
    edges = np.array([index, s_n, t_n])
    return (
        edges,
        input_data,
        sim_dat,
        sim_dat_aux,
        vessel_name,
        sim_dat_const,
        sim_dat_const_aux,
    )


@jaxtyped(typechecker=typechecker)
def compute_beta(a_0: NDArray, h_0: float, x: float, vessel: dict) -> NDArray:
    """
    Computes the beta value for the vessel.

    Parameters:
    a_0 (NDArray): Initial cross-sectional area.
    h_0 (float): Initial thickness.
    e (float): Young's modulus.

    Returns:
    float: Computed beta value.
    """
    if "beta" in vessel:
        return [vessel["beta"]] * len(a_0)
    elif "beta_p" and "beta_s" in vessel:
        beta_p = vessel["beta_p"]
        beta_s = vessel["beta_s"]
        return np.array(beta_p + beta_s * np.arange(0, x, len(a_0)))

    elif "E" in vessel:
        e = float(vessel["E"])
        s_pi = np.sqrt(np.pi)
        s_pi_e_over_sigma_squared = s_pi * e / 0.75
        return np.array(1 / np.sqrt(a_0) * h_0 * s_pi_e_over_sigma_squared)
    else:
        exception_message = "Missing Young's modulus value for vessel"
        raise ValueError(exception_message)


@jaxtyped(typechecker=typechecker)
def compute_radius_slope(r_p: float, r_d: float, length: float) -> float:
    """
    Computes the radius slope of the vessel.

    Parameters:
    r_p (float): Proximal radius.
    r_d (float): Distal radius.
    length (float): Length of the vessel.

    Returns:
    float: Radius slope.
    """
    return (r_d - r_p) / length


@jaxtyped(typechecker=typechecker)
def compute_thickness(r_0_i: float) -> float:
    """
    Computes the thickness of the vessel wall.

    Parameters:
    r_0_i (float): Initial radius.

    Returns:
    float: Computed thickness.
    """
    a = 0.2802
    b = -5.053e2
    c = 0.1324
    d = -0.1114e2
    return r_0_i * (a * np.exp(b * r_0_i) + c * np.exp(d * r_0_i))


@jaxtyped(typechecker=typechecker)
def compute_radii(vessel: dict) -> tuple[float, float]:
    """
    Computes the proximal and distal radii of the vessel.

    Parameters:
    vessel (dict): Vessel configuration data.

    Returns:
    tuple[float, float]: Proximal and distal radii.
    """
    if "R0" not in vessel:
        r_p = float(vessel["Rp"])
        r_d = float(vessel["Rd"])
        return r_p, r_d
    else:
        r_0 = float(vessel["R0"])
        return r_0, r_0


@jaxtyped(typechecker=typechecker)
def get_pext(vessel: dict) -> float:
    """
    Gets the external pressure of the vessel.

    Parameters:
    vessel (dict): Vessel configuration data.

    Returns:
    float: External pressure.
    """
    if "pext" not in vessel:
        return 0.0
    else:
        return vessel["pext"]


@jaxtyped(typechecker=typechecker)
def get_phi(vessel: dict) -> float:
    """
    Gets the phi value of the vessel.

    Parameters:
    vessel (dict): Vessel configuration data.

    Returns:
    float: Phi value.
    """
    if "phi" not in vessel:
        return 0.0
    else:
        return vessel["phi"]


@jaxtyped(typechecker=typechecker)
def mesh_vessel(vessel: dict, length: float) -> float:
    """
    Computes the number of mesh points for the vessel.

    Parameters:
    vessel (dict): Vessel configuration data.
    length (float): Length of the vessel.

    Returns:
    int: Number of mesh points.
    """
    if "M" not in vessel:
        m = max([5, int(np.ceil(length * 1e3))])
    else:
        m = vessel["M"]
        m = max([5, m, int(np.ceil(length * 1e3))])

    return m


@jaxtyped(typechecker=typechecker)
def initialise_thickness(vessel: dict) -> float:
    """
    Initializes the thickness of the vessel wall.

    Parameters:
    vessel (dict): Vessel configuration data.

    Returns:
    float: Initial thickness.
    """
    if "h0" not in vessel:
        return 0.0
    else:
        return vessel["h0"]


@jaxtyped(typechecker=typechecker)
def add_outlet(vessel: dict) -> tuple[int, float, float, float, float]:
    """
    Adds the outlet configuration to the vessel.

    Parameters:
    vessel (dict): Vessel configuration data.

    Returns:
    tuple[int, float, float, float, float]: Outlet type and associated parameters.
    """
    outlet = 0
    r_t = 0.0
    r_1 = 0.0
    r_2 = 0.0
    c = 0.0
    if "outlet" in vessel:
        outlet = vessel["outlet"]
        if outlet == 3:  # "wk3"
            r_t = 0.0
            r_1 = float(vessel["R1"])
            r_2 = float(vessel["R2"])
            c = float(vessel["Cc"])
        elif outlet == 2:  # "wk2"
            r_t = 0.0
            r_1 = 0.0
            r_2 = float(vessel["R1"])
            c = float(vessel["Cc"])
        elif outlet == 1:  # "reflection"
            r_t = float(vessel["Rt"])
            r_1 = 0.0
            r_2 = 0.0
            c = 0.0

    return outlet, r_t, r_1, r_2, c


@jaxtyped(typechecker=typechecker)
def compute_viscous_term(vessel_data: dict, blood: Blood) -> float:
    """
    Computes the viscous term for the vessel.

    Parameters:
    vessel_data (dict): Vessel configuration data.
    blood (Blood): Blood object with specified properties.

    Returns:
    float: Viscous term.
    """
    gamma_profile = vessel_data.get("gamma_profile", 9)
    return 2 * (gamma_profile + 2) * np.pi * blood.mu * blood.rho_inv


@jaxtyped(typechecker=typechecker)
def build_heart(vessel_data: dict) -> tuple[bool, float, NDArray]:
    """
    Builds the heart data for the inlet vessel.

    Parameters:
    vessel_data (dict): Vessel configuration data.

    Returns:
    tuple[bool, float, NDArray]: Inlet status, cardiac period, and input data.
    """
    if "inlet" in vessel_data:
        input_data = np.loadtxt(vessel_data["inlet file"])
        cardiac_period = input_data[-1, 0]
        return True, cardiac_period, input_data
    else:
        return False, 0.0, np.zeros((1, 2))


@jaxtyped(typechecker=typechecker)
def compute_windkessel_inlet_impedance(
    r2: float, blood: Blood, a0: NDArray[np.floating[Any]], gamma: Array
) -> tuple[float, float]:
    """
    Computes the inlet impedance for the Windkessel model.

    Parameters:
    r2 (float): Windkessel resistance.
    blood (Blood): Blood object with specified properties.
    a0 (NDArray[np.floating[Any]]): Initial cross-sectional area.
    gamma (Array): Gamma value for wave speed calculation.

    Returns:
    tuple[float, float]: Updated Windkessel resistances.
    """
    r1 = blood.rho * wave_speed(a0[-1], gamma[-1]) / a0[-1]
    r2 -= r1

    return r1, r2
