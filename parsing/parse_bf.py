import os
import sys
import csv
from ruamel.yaml import YAML
import ruamel.yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(__file__))

modelname = sys.argv[1]
modelfilename = sys.argv[2]
model_sub_dir = ""
if len(sys.argv) > 3: 
    model_sub_dir = "/" + sys.argv[3]

models_dir = "openBF-hub/models"
model_dir = "openBF-hub/models/" + modelname  + model_sub_dir
csv_file = model_dir + "/" + modelfilename + '.csv'
test_path = "../test/" + modelfilename 
yaml_file = test_path + "/" + modelfilename + '.yml'
if os.path.exists(csv_file):
    if not os.path.exists(test_path): 
        os.mkdir(test_path)
    shutil.copyfile(model_dir + "/" + modelfilename + "_inlet.dat", test_path+ "/" + modelfilename + "_inlet.dat")
else:
    model_sub_dir = "/" + sys.argv[2]
    model_dir = "openBF-hub/models/" + modelname  + model_sub_dir
    csv_file = model_dir + "/" + modelfilename + '.csv'
    test_path = "../test/" + modelfilename 
    yaml_file = test_path + "/" + modelfilename + '.yml'
    if not os.path.exists(test_path): 
        os.mkdir(test_path)
    shutil.copyfile(model_dir + "/" + modelfilename + "_inlet.dat", test_path+ "/" + modelfilename + "_inlet.dat")



# Read the constants from file
constants = {}
with open(model_dir + "/" + modelfilename +'_constants.jl', 'r') as file:
    for line in file:
        parts = line.strip().split('=')
        key = parts[0].strip().replace('const ', '')
        value = parts[1].strip()
        constants[key] = value

# Create an OrderedDict for the YAML structure
data = ruamel.yaml.comments.CommentedMap()

# Add project name
data['project name'] = modelfilename

# Add blood section
blood = ruamel.yaml.comments.CommentedMap()
blood['rho'] = float(constants['rho'])
blood['mu'] = float(constants['mu'])
data['blood'] = blood

# Add solver section
solver = ruamel.yaml.comments.CommentedMap()
solver['Ccfl'] = float(constants['Ccfl'])
solver['cycles'] = int(constants['cycles'])
solver['num_snapshots'] = 100
solver['convergence tolerance'] = 1.0
data['solver'] = solver

# Create a YAML instance and configure the formatting
#yaml = ruamel.yaml.YAML()
#yaml.indent(mapping=2, sequence=4, offset=2)
#
## Write the YAML data to a file
#with open('adan56.yml', 'w') as file:
#    yaml.dump(data, file)


data_net = []
RCR = []

# Read the CSV file
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file, delimiter = ",", skipinitialspace=True)
    for row in reader:
        if int(row['sn']) == 1:
            entry = {
                    'label': row['Name'],
                    'sn': int(row['sn']),
                    'tn': int(row['tn']),
                    'L': float(row['l(m)']),
                    'R0': float(row['Rp(m)']),
                    'E': float(row['E(Pa)']),
                    'inlet': 1 if constants['inlet_type'].replace('"', '') == "Q" else 2,
                    'inlet file': "test/" + modelfilename + "/" + modelfilename + '_inlet.dat',
                    'inlet number': int(row['wkn'])
                }
            data_net.append(entry)

        elif float(row['R1']) > 0 and 'R2' in row:
            entry = {
                    'label': row['Name'],
                    'sn': int(row['sn']),
                    'tn': int(row['tn']),
                    'L': float(row['l(m)']),
                    'R0': float(row['Rp(m)']),
                    'E': float(row['E(Pa)']),
                    'outlet': 3,
                    'R1': float(row['R1']),
                    'R2': float(row['R2']),
                    'Cc': float(row['C']),
                }
            data_net.append(entry)
            RCR.append([float(row['R1']), float(row['C']), float(row['R2'])])
        elif float(row['R1']) > 0:
            entry = {
                    'label': row['Name'],
                    'sn': int(row['sn']),
                    'tn': int(row['tn']),
                    'L': float(row['l(m)']),
                    'R0': float(row['Rp(m)']),
                    'E': float(row['E(Pa)']),
                    'outlet': 2,
                    'R1': float(row['R1']),
                    'Cc': float(row['C']),
                }
            data_net.append(entry)
        else:
            entry = {
                    'label': row['Name'],
                    'sn': int(row['sn']),
                    'tn': int(row['tn']),
                    'L': float(row['l(m)']),
                    'R0': float(row['Rp(m)']),
                    'E': float(row['E(Pa)']),
                }
            data_net.append(entry)

data['network'] = data_net
# Write the YAML output
yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
with open(yaml_file, 'w') as file:
    yaml.dump(data, file)


#index = np.array(range(len(RCR)))
#print(index)
#print(np.array(RCR)[:,0])
#plt.figure()
#plt.plot(index,np.array(RCR)[:,0])
#plt.plot(index,np.array(RCR)[:,1])
#plt.plot(index,np.array(RCR)[:,2])
#plt.show()