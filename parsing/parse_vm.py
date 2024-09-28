import numpy as np
import sys
from ruamel.yaml import YAML
import ruamel.yaml
import os
from bs4 import BeautifulSoup

# Set the working directory to the directory of the script
os.chdir(os.path.dirname(__file__))


def compute_thickness(r0):
    """
    Compute the vessel wall thickness based on the initial radius (r0).

    Args:
        r0 (float): Initial radius of the vessel.

    Returns:
        float: Computed wall thickness of the vessel.
    """
    ah = 0.2802
    bh = -5.053e2
    ch = 0.1324
    dh = -0.1114e2
    return r0 * (ah * np.exp(bh * r0) + ch * np.exp(dh * r0))


# Load command line argument (model name) and setup file paths
modelname = sys.argv[1]
clinical_dat_filename = f"models_vm/{modelname}/{modelname}_ClinicalData.csv"
casename = open(clinical_dat_filename).readlines()[2].split(",")[0]
path = f"../test/{modelname}"
yaml_file = f"{path}/{modelname}.yml"

# Create output directory if it doesn't exist
if not os.path.exists(path):
    os.mkdir(path)

# Process inflow data
inflow_const = 1e4
inflow = np.loadtxt(
    "models_vm/" + modelname + "/Simulations/" + casename + "/inflow.flow"
)
inflow[:, 1] = inflow[:, 1] / inflow_const
np.savetxt(path + "/inflow.flow", inflow)

# Load network properties
tsv_filename = "models_vm/" + modelname + "/network_properties.tsv"
dat = np.loadtxt(tsv_filename, skiprows=1)
N = dat.shape[0]
dat_edges = dat[:, -6:]
dat_edges_seq = np.zeros((2 * N, 3))
dat_edges_seq[:N, :] = dat_edges[:, -6:-3]
dat_edges_seq[N : 2 * N, :] = dat_edges[:, -3:]
dat_edges_seq_tuples = [tuple(row) for row in dat_edges_seq]
dat_edges_seq_unique = np.unique(dat_edges_seq_tuples, axis=0)
dat_edges_seq_unique = [tuple(row) for row in dat_edges_seq_unique]

# Create YAML structure
data = ruamel.yaml.comments.CommentedMap()
data["proj_name"] = modelname

# Add blood properties
blood = ruamel.yaml.comments.CommentedMap()
blood["rho"] = 1060.0
blood["mu"] = 0.004
data["blood"] = blood

# Add solver properties
solver = ruamel.yaml.comments.CommentedMap()
solver["Ccfl"] = 0.9
solver["num_snapshots"] = 100
solver["conv_tol"] = 1.0
data["solver"] = solver

# Set up network data arrays
data_net = []
sn = np.zeros(N, dtype=np.int64)
tn = np.zeros(N, dtype=np.int64)

# Identify and assign node IDs (start and terminal nodes)
for i in range(0, N):
    sn[i] = np.where(dat[i, 6:9] == dat_edges_seq_unique)[0][0] + 1
    tn[i] = np.where(dat[i, 9:12] == dat_edges_seq_unique)[0][0] + 1

temp = np.zeros(2 * N, dtype=np.int64)
temp[0:N] = sn
temp[N : 2 * N] = tn
unique, counts = np.unique(temp, return_counts=True)

# Read simulation path data
with open(
    f"models_vm/{modelname}/Simulations/{casename}.sjb",
    "r",
) as f:
    path_data = f.readlines()[2:]
path_data = "".join(line for line in path_data)

# Extract vessel and outlet names
directory = f"models_vm/{modelname}/Paths"
vessel_names = [
    filename.split(".")[0]
    for filename in os.listdir(directory)
    if os.path.isfile(os.path.join(directory, filename))
]

outlet_names = []
RCR = []
for vessel_name in vessel_names:
    bs_data = BeautifulSoup(path_data, "xml").find("cap", {"name": vessel_name})
    if bs_data is not None:
        outlet_names.append(bs_data["name"])
        bc_type = bs_data.find("prop", {"key": "BC Type"})["value"]
        values = bs_data.find("prop", {"key": "Values"})["value"].split(" ")

        # Determine boundary condition type and compute RCR values
        if bc_type == "Resistance":
            Rt = values[0]
            RCR.append([Rt])
        elif bc_type == "Coronary":
            R1 = float(values[0]) * 1e5
            Cc = float(values[1]) * 1e-5
            R2 = float(values[2]) * 1e5
            RCR.append([R1, Cc, R2])
        else:
            R1 = float(values[0]) * 1e5
            Cc = float(values[1]) * 1e-5
            R2 = float(values[2]) * 1e5
            RCR.append([R1, Cc, R2])

# Match points between vessels and outlets based on geometry
V = len(vessel_names)
W = len(outlet_names)
match_points = np.zeros((2 * W, 3))
counter = 0
for filename in os.listdir(directory):
    if any(vessel_names[i] == s for s in outlet_names):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            with open(
                f,
                "r",
            ) as f:
                path_data = f.readlines()[2:]
            path_data = "".join(line for line in path_data)
            bs_data = (
                BeautifulSoup(path_data, "xml").find("control_points").find_all("point")
            )
            match_points[2 * counter, :] = [
                float(bs_data[0]["x"]),
                bs_data[0]["y"],
                -float(bs_data[0]["z"]),
            ]
            match_points[2 * counter + 1, :] = [
                float(bs_data[-1]["x"]),
                bs_data[-1]["y"],
                -float(bs_data[-1]["z"]),
            ]
            counter = counter + 1

# Determine outlet potential names based on closest geometric matches
outlet_pot_names = []
if match_points.size > 0:
    for i in range(0, N):
        j = np.argmin(np.linalg.norm(match_points + dat_edges[i, -3:], axis=1))
        outlet_pot_names.append(outlet_names[int(j / 2)])


# Reorder network data and process inlet/outlet properties
temp = np.zeros((N, 2))
temp[:, 0] = sn
temp[:, 1] = tn
args = temp[:, 0].argsort()
temp = temp[args]
sn = temp[:, 0]
tn = temp[:, 1]
dat = dat[args, :]
counts = dict(zip(unique, counts))
inlet_counter = 1
outlet_counter = 0

# Compute network properties and add entries to YAML data structure
k1, k2, k3 = 2.0e6, -2253.00, 86500.00
for i in range(N):
    if counts[sn[i]] == 1:
        # compute Young's modulus as in
        # Applied Mathematical Models in Human Physiology p109
        r0 = dat[i, 2] / 100
        h0 = compute_thickness(r0)
        E = r0 / h0 * (k1 * np.exp(k2 * r0) + k3)

        name = outlet_pot_names[i] if outlet_pot_names else f"vessel{i}"
        entry = {
            "label": name,
            "sn": int(sn[i]),
            "tn": int(tn[i]),
            "L": float(dat[i, 1] / 100),
            "R0": float(r0),
            "E": float(E),
            "inlet": int(1),
            "inlet file": "test/" + modelname + "/inflow.flow",
            "inlet number": int(inlet_counter),
        }
        inlet_counter += 1
        data_net.append(entry)

inlet_counter = 1
outlet_counter = 0
R_tot = 120 * 133.32 / 8e-5
A_tot = 0.0
Cc = 1e-8
for i in range(N):
    if counts[tn[i]] == 1:
        r0 = dat[i, 2] / 100
        A_tot = A_tot + r0 * r0 * np.pi

R1_temp = []
R2_temp = []
Cc_temp = []
for i in range(0, N):
    name = outlet_pot_names[i] if outlet_pot_names else f"vessel{i}"
    r0 = dat[i, 2] / 100
    h0 = compute_thickness(r0)
    E = r0 / h0 * (k1 * np.exp(k2 * r0) + k3)
    if counts[sn[i]] == 1:
        print(" ")

    elif counts[tn[i]] == 1:
        R_const = 2
        R_i = A_tot / (r0 * r0 * np.pi) * R_tot
        R1 = 0.09 * R_i * R_const
        R2 = 0.91 * R_i * R_const
        Cc_i = Cc * (r0 * r0 * np.pi) / A_tot
        R1_temp.append(R1)
        R2_temp.append(R2)
        Cc_temp.append(Cc_i)
        entry = {
            "label": name,
            "sn": int(sn[i]),
            "tn": int(tn[i]),
            "L": float(dat[i, 1] / 100),
            "R0": float(r0),
            "E": float(E),
            "outlet": int(3),
            "R1": float(R1),
            "R2": float(R2),
            "Cc": float(Cc_i),
        }
        outlet_counter = outlet_counter + 1
        data_net.append(entry)
    else:
        entry = {
            "label": name,
            "sn": int(sn[i]),
            "tn": int(tn[i]),
            "L": float(dat[i, 1] / 100),
            "R0": float(r0),
            "E": float(E),
        }
        data_net.append(entry)

# Add network data to YAML structure
data["network"] = data_net

# Write the YAML data to the output file
yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
with open(yaml_file, "w") as file:
    yaml.dump(data, file)
