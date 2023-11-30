import os
import sys
import csv
from ruamel.yaml import YAML
import ruamel.yaml
import shutil
os.chdir(os.path.dirname(__file__))

modelname = sys.argv[1]
models_dir = "openBF-hub/models"
model_dir = "openBF-hub/models/" + modelname
csv_file = model_dir + "/" + modelname + '.csv'
test_path = "../test/" + modelname 
yaml_file = test_path + "/" + modelname + '.yml'
if not os.path.exists(test_path): 
    os.mkdir(test_path)
shutil.copyfile(model_dir + "/" + modelname + "_inlet.dat", test_path+ "/" + modelname + "_inlet.dat")


# Read the constants from file
constants = {}
with open(model_dir + "/" + modelname +'_constants.jl', 'r') as file:
    for line in file:
        parts = line.strip().split('=')
        key = parts[0].strip().replace('const ', '')
        value = parts[1].strip()
        constants[key] = value

# Create an OrderedDict for the YAML structure
data = ruamel.yaml.comments.CommentedMap()

# Add project name
data['project name'] = modelname

# Add blood section
blood = ruamel.yaml.comments.CommentedMap()
blood['rho'] = float(constants['rho'])
blood['mu'] = float(constants['mu'])
data['blood'] = blood

# Add solver section
solver = ruamel.yaml.comments.CommentedMap()
solver['Ccfl'] = float(constants['Ccfl'])
solver['cycles'] = int(constants['cycles'])
solver['jump'] = 100
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

# Read the CSV file
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file, delimiter = ",", skipinitialspace=True)
    for row in reader:
        print(row)
        if int(row['sn']) == 1:
            entry = {
                    'label': row['Name'],
                    'sn': int(row['sn']),
                    'tn': int(row['tn']),
                    'L': float(row['l(m)']),
                    'R0': float(row['Rp(m)']),
                    'E': float(row['E(Pa)']),
                    'inlet': constants['inlet_type'].replace('"', ''),
                    'inlet file': "test/" + modelname + "/" + modelname + '_inlet.dat',
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
                    'outlet': 'wk3',
                    'R1': float(row['R1']),
                    'R2': float(row['R2']),
                    'Cc': float(row['C']),
                }
            data_net.append(entry)
        elif float(row['R1']) > 0:
            entry = {
                    'label': row['Name'],
                    'sn': int(row['sn']),
                    'tn': int(row['tn']),
                    'L': float(row['l(m)']),
                    'R0': float(row['Rp(m)']),
                    'E': float(row['E(Pa)']),
                    'outlet': 'wk2',
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
print(data)
with open(yaml_file, 'w') as file:
    yaml.dump(data, file)

