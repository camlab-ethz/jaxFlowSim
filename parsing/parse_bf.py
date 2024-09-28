import os
import sys
import csv
from ruamel.yaml import YAML
import ruamel.yaml
import shutil

# Set the working directory to the directory of the script
os.chdir(os.path.dirname(__file__))

# Get command-line arguments
modelname = sys.argv[1]  # Model name
modelfilename = sys.argv[2]  # Model file name
model_sub_dir = ""  # Optional subdirectory
if len(sys.argv) > 3:
    model_sub_dir = "/" + sys.argv[3]

# Define directories and file paths
models_dir = "openBF-hub/models"
model_dir = f"openBF-hub/models/{modelname}{model_sub_dir}"
csv_file = f"{model_dir}/{modelfilename}.csv"
test_path = f"../test/{modelfilename}"
yaml_file = f"{test_path}/{modelfilename}.yml"

# Check if the CSV file exists, if not, modify model subdirectory
if os.path.exists(csv_file):
    # Create test path directory if it doesn't exist and copy inlet file
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    shutil.copyfile(
        f"{model_dir}/{modelfilename}_inlet.dat",
        f"{test_path}/{modelfilename}_inlet.dat",
    )
else:
    model_sub_dir = "/" + sys.argv[2]  # Update subdirectory
    model_dir = f"openBF-hub/models/{modelname}{model_sub_dir}"
    csv_file = f"{model_dir}/{modelfilename}.csv"
    test_path = f"../test/{modelfilename}"
    yaml_file = f"{test_path}/{modelfilename}.yml"
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    shutil.copyfile(
        f"{model_dir}/{modelfilename}_inlet.dat",
        f"{test_path}/{modelfilename}_inlet.dat",
    )

# Read constants from the constants file
constants = {}
with open(f"{model_dir}/{modelfilename}_constants.jl", "r") as file:
    for line in file:
        parts = line.strip().split("=")
        key = parts[0].strip().replace("const ", "")
        value = parts[1].strip()
        constants[key] = value

# Create an OrderedDict (CommentedMap) for the YAML structure
data = ruamel.yaml.comments.CommentedMap()

# Add project name to the YAML
data["proj_name"] = modelfilename

# Add blood properties to the YAML
blood = ruamel.yaml.comments.CommentedMap()
blood["rho"] = float(constants["rho"])
blood["mu"] = float(constants["mu"])
data["blood"] = blood

# Add solver properties to the YAML
solver = ruamel.yaml.comments.CommentedMap()
solver["Ccfl"] = float(constants["Ccfl"])  # CFL condition number
solver["num_snapshots"] = 100  # Number of snapshots for solver
solver["conv_tol"] = 1.0  # Convergence tolerance
data["solver"] = solver

# Initialize lists to hold network and RCR (resistor-capacitor-resistor) values
data_net = []
RCR = []

# Read the CSV file and process vessel properties
with open(csv_file, "r") as file:
    reader = csv.DictReader(file, delimiter=",", skipinitialspace=True)
    for row in reader:
        # If the vessel is an inlet
        if int(row["sn"]) == 1:
            entry = {
                "label": row["Name"],
                "sn": int(row["sn"]),
                "tn": int(row["tn"]),
                "L": float(row["l(m)"]),
                "R0": float(row["Rp(m)"]),
                "E": float(row["E(Pa)"]),
                "inlet": 1 if constants["inlet_type"].replace('"', "") == "Q" else 2,
                "inlet file": f"test/{modelfilename}/{modelfilename}_inlet.dat",
                "inlet number": int(row["wkn"]),
            }
            data_net.append(entry)

        # If the vessel has an RCR outlet
        elif float(row["R1"]) > 0 and "R2" in row:
            entry = {
                "label": row["Name"],
                "sn": int(row["sn"]),
                "tn": int(row["tn"]),
                "L": float(row["l(m)"]),
                "R0": float(row["Rp(m)"]),
                "E": float(row["E(Pa)"]),
                "outlet": 3,
                "R1": float(row["R1"]),
                "R2": float(row["R2"]),
                "Cc": float(row["C"]),
            }
            data_net.append(entry)
            RCR.append([float(row["R1"]), float(row["C"]), float(row["R2"])])

        # If the vessel has a 2-parameter Windkessel outlet
        elif float(row["R1"]) > 0:
            entry = {
                "label": row["Name"],
                "sn": int(row["sn"]),
                "tn": int(row["tn"]),
                "L": float(row["l(m)"]),
                "R0": float(row["Rp(m)"]),
                "E": float(row["E(Pa)"]),
                "outlet": 2,
                "R1": float(row["R1"]),
                "Cc": float(row["C"]),
            }
            data_net.append(entry)

        # If it's a normal vessel
        else:
            entry = {
                "label": row["Name"],
                "sn": int(row["sn"]),
                "tn": int(row["tn"]),
                "L": float(row["l(m)"]),
                "R0": float(row["Rp(m)"]),
                "E": float(row["E(Pa)"]),
            }
            data_net.append(entry)

# Add network data to the YAML
data["network"] = data_net

# Write the YAML output
yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)  # Configure YAML formatting
with open(yaml_file, "w") as file:
    yaml.dump(data, file)
