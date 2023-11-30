import sys
import os
import numpy as np
from ruamel.yaml import YAML
import ruamel.yaml
os.chdir(os.path.dirname(__file__))

def computeThickness(R0i):
    ah = 0.2802
    bh = -5.053e2
    ch = 0.1324
    dh = -0.1114e2
    return R0i * (ah * np.exp(bh * R0i) + ch * np.exp(dh * R0i))

modelname = sys.argv[1]
model_dir = "models_gk/" + modelname
model_filename = model_dir + "/" + modelname + ".in"
path = "../test/"+ modelname
yaml_filename = path +"/" + modelname + ".yml" 
if not os.path.exists(path): 
    os.mkdir(path)


rho = 0.0
mu = 0.0
Ccfl = 0.0
cycles = 100
jump = 100
convergence_tolerance = 1.0
Pext = 0.0
N = 0
sec_bc = False
sec_ves = False
inputs = []
T = 0.0
M = 200
temp_A = 0.0
sn = 0
tn = 0 
r = 0.0
L = 0.0
R1 = []
Cc = []
edges_counter = 0
domain_bc = 0
domain_ves = 0
outlet_counter = 1
outlet_type = 0
nodes = []
s_nodes = []
t_nodes = []

with open(model_filename) as file:
    i = 0
    for line in file:
        if "Rho" in line:
            rho = float(line.split("\t")[0])
        if "Viscosity" in line:
            mu = float(line.split("\t")[0])
        if "pext" in line:
            Pext = float(line.split("\t")[0])
        if "T (s)" in line:
            T = float(line.split("\t")[0])
        if "domain" in line and sec_ves:
            temp_line = line.split(" ")
            temp_line = [x for x in temp_line if x!='']
            k = temp_line.index("domain")
            domain_ves = int(temp_line[k+1].replace("\n", ""))
        if "Ndomains" in line:
            N = int(line.split(" ")[-1])
            sn = np.zeros(N, dtype=np.int64)
            tn = np.zeros(N, dtype=np.int64)
            s_nodes = np.zeros((N,3), dtype=np.int64)
            t_nodes = np.zeros((N,3), dtype=np.int64)
            outlet_type = np.zeros(N, dtype=np.int64)
            r = np.zeros(N, dtype=np.float64)
            L = np.zeros(N, dtype=np.float64)
            R1 = np.zeros(N, dtype=np.float64)
            Cc = np.zeros(N, dtype=np.float64)
            sec_ves = True
        if "Ao" in line and sec_ves:
            temp_line = line.split(" ")
            temp_line = [x for x in temp_line if x!='']
            temp_line = [x for x in temp_line if x!='\n']
            A0 = float(temp_line[-1].replace("=", "").replace("\n", ""))
            r0 = np.sqrt(A0/np.pi)
            r[domain_ves-1] = r0
        if "x_upper" in line and sec_ves:
            temp_line = line.split(" ")
            temp_line = [x for x in temp_line if x!='']
            l = float(temp_line[1])
            L[domain_ves-1] = l
        if "Boundary conditions" in line:
            sec_ves = False
            sec_bc = True
        if "Initial conditions" in line:
            sec_bc = False
        
        
        if "Domain" in line and sec_bc:
            line1 = file.readline()
            line2 = file.readline()
            line3 = file.readline()
            temp_line = line.split(" ")
            temp_line = [x for x in temp_line if x!='']
            k = temp_line.index("Domain")
            domain_bc = int(temp_line[k+1].replace("\n", ""))
            if not "boundary Domain" in line or "boundary   Domain" in line:
                #val1 = int(temp_line[1])-1
                #val2 = int(temp_line[2])-1
                #tn[val1] = domain_bc
                #tn[val2] = domain_bc
                if "Compliance" in line2 and sec_bc: 
                    temp_line = line2.split(" ")
                    temp_line = [x for x in temp_line if x!='']
                    Cc[domain_bc-1] = float(temp_line[1])
                if "Resistance" in line3 and sec_bc: 
                    temp_line = line.split(" ")
                    temp_line = [x for x in temp_line if x!='']
                    val1 = int(temp_line[1])
                    val2 = int(temp_line[2])
                    if val1 != val2:
                        nodes.append(np.array(sorted([domain_bc, val1, val2])))
                        s_nodes[domain_bc-1,:] = np.array(sorted([domain_bc, val1, val2]))
                    else:
                        nodes.append(np.array(sorted([domain_bc, val1])))
                        s_nodes[domain_bc-1,:2] = np.array(sorted([domain_bc, val1]))
                    temp_line = line3.split(" ")
                    temp_line = [x for x in temp_line if x!='']
                    R1[domain_bc-1] = float(temp_line[1])
                    outlet_counter = outlet_counter+1
                    if "R" in line3:
                        outlet_type[domain_bc-1] = 1
                    if "W" in line3:
                        outlet_type[domain_bc-1] = 2
                else:
                    temp_line = line.split(" ")
                    temp_line = [x for x in temp_line if x!='']
                    print(temp_line)
                    val1 = int(temp_line[1])
                    val2 = int(temp_line[2])
                    if val1 != val2:
                        nodes.append(np.array(sorted([domain_bc, val1, val2])))
                        s_nodes[domain_bc-1,:] = np.array(sorted([domain_bc, val1, val2]))
                    else:
                        nodes.append(np.array(sorted([domain_bc, val1])))
                        s_nodes[domain_bc-1,:2] = np.array(sorted([domain_bc, val1]))
                    temp_line = line2.split(" ")
                    temp_line = [x for x in temp_line if x!='']
                    val1 = int(temp_line[1])
                    val2 = int(temp_line[2])
                    if val1 != val2:
                        nodes.append(np.array(sorted([domain_bc, val1, val2])))
                        t_nodes[domain_bc-1,:] = np.array(sorted([domain_bc, val1, val2]))
                    else:
                        nodes.append(np.array(sorted([domain_bc, val1])))
                        t_nodes[domain_bc-1,:2] = np.array(sorted([domain_bc, val1]))
            elif "q = " in line3 and sec_bc: 
                input = np.zeros((M,2)) 
                input[:,0] = np.arange(0,T,T/M)
                for i in range(M):
                    line_t = line3.replace("t", str(input[i,0]))
                    line_t = line_t.replace("cos", "np.cos")
                    line_t = line_t.replace("sin", "np.sin")
                    line_t = line_t.replace("   ", "")
                    line_t = line_t.replace(" ", "")
                    line_t = line_t.replace("=", "")
                    line_t = line_t.replace("q", "")
                    input[i,1] = eval(line_t)

                inputs.append(input)
                line4 = file.readline()
                temp_line = line4.split(" ")
                temp_line = [x for x in temp_line if x!='']
                val1 = int(temp_line[1])
                val2 = int(temp_line[2])
                if val1 != val2:
                    nodes.append(np.array(sorted([domain_bc, val1, val2])))
                    t_nodes[domain_bc-1,:] = np.array(sorted([domain_bc, val1, val2]))
                else:
                    nodes.append(np.array(sorted([domain_bc, val1])))
                    t_nodes[domain_bc-1,:2] = np.array(sorted([domain_bc, val1]))
        
            elif "A = " in line1 and sec_bc: 
                temp_A = float(line1.split(" ")[5])
                input = np.zeros((M,2)) 
                input[:,0] = np.arange(0,T,T/M)
                for i in range(M):
                    line_t = line3.replace("t", str(input[i,0]))
                    line_t = line_t.replace("cos", "np.cos")
                    line_t = line_t.replace("sin", "np.sin")
                    line_t = line_t.replace("   ", "")
                    line_t = line_t.replace(" ", "")
                    line_t = line_t.replace("=", "")
                    line_t = line_t.replace("u", "")
                    input[i,1] = eval(line_t)*temp_A

                inputs.append(input)
                line4 = file.readline()
                temp_line = line4.split(" ")
                temp_line = [x for x in temp_line if x!='']
                val1 = int(temp_line[1])
                val2 = int(temp_line[2])
                if val1 != val2:
                    nodes.append(np.array(sorted([domain_bc, val1, val2])))
                    t_nodes[domain_bc-1,:] = np.array(sorted([domain_bc, val1, val2]))
                else:
                    nodes.append(np.array(sorted([domain_bc, val1])))
                    t_nodes[domain_bc-1,:2] = np.array(sorted([domain_bc, val1]))



        #if "lhs boundary Bifur" in line and sec_bc: 
        #    print(line)
        #    if "A" in line:
        #        temp_line = line.split(" ")
        #        temp_line = [x for x in temp_line if x!='']
        #        sn[domain_bc-1] = int(temp_line[1])*N + int(temp_line[2])
        #        print(sn[domain_bc-1])

        #if "rhs boundary Bifur" in line and sec_bc: 
        #    print(line)
        #    if "A" in line:
        #        temp_line = line.split(" ")
        #        temp_line = [x for x in temp_line if x!='']
        #        tn[domain_bc-1] = int(temp_line[1])*N + int(temp_line[2])
        #        print(tn[domain_bc-1])
        #if "Compliance" in line and sec_bc: 
        #    temp_line = line.split(" ")
        #    temp_line = [x for x in temp_line if x!='']
        #    Cc[domain_bc-1] = float(temp_line[1])
        #if "Resistance" in line and sec_bc: 
        #    temp_line = line.split(" ")
        #    temp_line = [x for x in temp_line if x!='']
        #    R1[domain_bc-1] = float(temp_line[1])

nodes = [arr.tolist() for arr in nodes]
nodes_unique = []
for item in nodes: 
    if item not in nodes_unique: 
        nodes_unique.append(item) 
print(nodes_unique)
print(s_nodes)
print(t_nodes)
s_nodes = [arr.tolist() for arr in s_nodes]
t_nodes = [arr.tolist() for arr in t_nodes]
print(len(t_nodes))
outlet_counter = 1
inlet_counter = 1
for i in range(N):
    
    try:
        if s_nodes[i][-1] != 0:
            sn[i] = nodes_unique.index(s_nodes[i])
        else:
            sn[i] = nodes_unique.index(s_nodes[i][:2])
    except ValueError:
        sn[i] = -inlet_counter
        inlet_counter = inlet_counter + 1
    try:
        if t_nodes[i][-1] != 0:
            tn[i] = nodes_unique.index(t_nodes[i])
        else:
            tn[i] = nodes_unique.index(t_nodes[i][:2])
    except ValueError:
        tn[i] = len(nodes_unique) - 1 + outlet_counter
        #tn[i] = N + outlet_counter
        outlet_counter = outlet_counter + 1
sn = sn + inlet_counter 
tn = tn + inlet_counter 
print(sn)
print(tn)
temp = np.zeros(2*N, dtype=np.int64)
temp[0:N] = sn
temp[N:2*N] = tn
unique, counts = np.unique(temp, return_counts=True)
                
h0 = computeThickness(r)
k1 = 2.e7*100
k2 = -22.53*100
k3 = 8.65e5*100
#k1 = 2.e6
#k2 = -2253.00
#k3 = 86500.00
E = r0/h0 * (k1*np.exp(k2*r0) + k3)/1000 #Applied Mathematical Models in Human Physiology p109
    
        
            
# Create an OrderedDict for the YAML structure
data = ruamel.yaml.comments.CommentedMap()

# Add project name
data['project name'] = modelname

# Add blood section
blood = ruamel.yaml.comments.CommentedMap()
blood['rho'] = rho
blood['mu'] = mu
data['blood'] = blood

# Add solver section
solver = ruamel.yaml.comments.CommentedMap()
solver['Ccfl'] = Ccfl
solver['cycles'] = cycles
solver['jump'] = jump
solver['convergence tolerance'] = convergence_tolerance
data['solver'] = solver

data_net = []
counts = dict(zip(unique, counts))
inlet_counter = 1
outlet_counter = 0
print(counts)
for i in range(0,N):
    if counts[sn[i]] == 1: 
        name = "vessel" + str(i)
        #R1 = (4*np.pi*scc.epsilon_0)*1009
        #R2 = (4*np.pi*scc.epsilon_0)*19171
        #Cc = 1/(4*np.pi*scc.epsilon_0)*5.580000000000001e-5
        #R1 = (1e5)*1009
        #R2 = (1e5)*19171
        #Cc = (1e-5)*5.580000000000001e-5
        inlet_filename = "test/" + modelname + "/inflow" + str(inlet_counter) + ".flow"
        entry = {
                 'label': name,
                    'sn': int(sn[i]),
                    'tn': int(tn[i]),
                    'L': float(L[i]),
                    'R0': float(r[i]),
                    'E': float(E[i]),
                    'inlet': int(1),
                    'inlet file': inlet_filename,
                    'inlet number': int(inlet_counter)
        }
        with open("../" + inlet_filename, "w") as file:
            np.savetxt(file, inputs[inlet_counter-1])
        inlet_counter = inlet_counter + 1
        data_net.append(entry)

inlet_counter = 1
outlet_counter = 0
for i in range(0,N):
    name = "vessel" + str(i)
    #R1 = (1e5)*1009
    #R2 = (1e5)*19171
    #Cc = (1e-5)*5.580000000000001e-5
    if counts[sn[i]] == 1: 
        print(" ")
    elif counts[tn[i]] == 1: 
        if outlet_type[i] == 1:
            entry = {
                        'label': name,
                        'sn': int(sn[i]),
                        'tn': int(tn[i]),
                        'L': float(L[i]),
                        'R0': float(r[i]),
                        'E': float(E[i]),
                        'outlet': int(outlet_type[i]),
                        'Rt': float(R1[i])
                        #'R1': R1,
                        #'R2': R2,
                        #'Cc': Cc,
            }
            outlet_counter = outlet_counter + 1
            data_net.append(entry)
        elif outlet_type[i] == 2:
            entry = {
                        'label': name,
                        'sn': int(sn[i]),
                        'tn': int(tn[i]),
                        'L': float(L[i]),
                        'R0': float(r[i]),
                        'E': float(E[i]),
                        'outlet': int(outlet_type[i]),
                        #'Rt': 0.0,
                        #'R1': 1863776174.7e-1,
                        #'R2': 4699455106.8-1,
                        #'Cc': 4.821160687e-10,
                        #'R1': 1863776174.7,
                        #'R2': 4699455106.8,
                        #'Cc': 4.821160687e-11,
                        #'R1': 1009e5,
                        #'R2': 19171e5,
                        #'Cc': 5.580000000000001e-10,
                        #'R1': 1009,
                        #'R2': 19171,
                        #'Cc': 5.580000000000001e-5,
                        'R1': float(R1[i]),
                        'Cc': float(Cc[i]),
                        #'R1': RCR[m][0],
                        #'R2': RCR[m][2],
                        #'Cc': RCR[m][1],
            }
            outlet_counter = outlet_counter + 1
            data_net.append(entry)

    else: 
        entry = {
                    'label': name,
                    'sn': int(sn[i]),
                    'tn': int(tn[i]),
                    'L': float(L[i]),
                    'R0': float(r[i]),
                    'E': float(E[i])
        }
        data_net.append(entry)

data_net.sort(key=lambda x: x["sn"])
data['network'] = data_net
# Write the YAML output
yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)
with open(yaml_filename, 'w') as file:
    yaml.dump(data, file)