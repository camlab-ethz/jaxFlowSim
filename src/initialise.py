import numpy as np
import yaml
import os.path
import shutil
from jax.tree_util import Partial
#from functools import partial
#from jax import jit
from src.utils import waveSpeed, pressureSA
from src.components import Blood
from src.anastomosis import solveAnastomosisWrapper
from src.bifurcations import solveBifurcationWrapper
from src.conjunctions import solveConjunctionWrapper
from src.boundary_conditions import setReflectionOutletBCWrapper, setWindkesselOutletBCWrapper

#JUNCTION_FUNCTIONS = []

def loadConfig(input_filename):
    data = loadYamlFile(input_filename)
    checkInputFile(data) 
    return data


def loadYamlFile(filename):
    if not os.path.isfile(filename):
        raise ValueError(f"missing file {filename}")

    with open(filename, "r") as file:
        return yaml.safe_load(file)


def checkInputFile(data):
    checkSections(data)
    checkNetwork(data["network"])


def checkSections(data):
    keys = ["proj_name", "network", "blood", "solver"]
    for key in keys:
        if key not in data:
            raise ValueError(f"missing section {key} in YAML input file")

    checkSection(data, "blood", ["mu", "rho"])
    checkSection(data, "solver", ["Ccfl", "conv_tol"])

    if "num_snapshots" not in data["solver"]:
        data["solver"]["num_snapshots"] = 100


def checkSection(data, section, keys):
    for key in keys:
        if key not in data[section]:
            raise ValueError(f"missing {key} in {section} section")


def checkNetwork(network):
    has_inlet = False
    inlets = set()
    has_outlet = False
    nodes = {}

    for i, vessel in enumerate(network, start=1):
        checkVessel(i, vessel)

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
                raise ValueError(f"outlet not defined for vessel {vessel['label']}, check connectivity")

    if not has_inlet:
        raise ValueError("missing inlet(s) definition")

    if not has_outlet:
        raise ValueError("missing outlet(s) definition")

def checkVessel(i, vessel):
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
                raise ValueError(f"outlet vessel {i} is missing three-element windkessel values")
        elif outlet == "wk2":
            if "R1" not in vessel or "Cc" not in vessel:
                raise ValueError(f"outlet vessel {i} is missing two-element windkessel values")
        elif outlet == "reflection":
            if "Rt" not in vessel:
                raise ValueError(f"outlet vessel {i} is missing reflection coefficient value")





def makeResultsFolder(data, input_filename):
    project_name = data["proj_name"]

    if "results folder" not in data:
        r_folder = "results/" +  project_name + "_results"
    else:
        r_folder = data["results folder"]

    # delete existing folder and results
    if os.path.isdir(r_folder):
        shutil.rmtree(r_folder)

    os.makedirs(r_folder, mode = 0o777)
    shutil.copy2(input_filename, r_folder + "/")
    copyInletFiles(data, r_folder)


def copyInletFiles(data, r_folder):
    for vessel in data["network"]:
        if "inlet file" in vessel:
            shutil.copy2(vessel["inlet file"], r_folder + "/")


def buildBlood(blood_data):
    mu = blood_data["mu"]
    rho = blood_data["rho"]
    rho_inv = 1.0 / rho

    return Blood(mu, rho, rho_inv)


def buildArterialNetwork(network, blood):
    
    B = 2
    N = len(network)
    M_0 = meshVessel(network[0], float(network[0]["L"]))
    starts = np.zeros(N, dtype=np.int64)
    ends = np.zeros(N, dtype=np.int64)

    starts[0] = B
    ends[0] = M_0 + B

    for i in range(1, N):
        L = float(network[i]["L"])
        _M = meshVessel(network[i], L)
        starts[i] = ends[i-1] + 2*B
        ends[i] = starts[i] + _M

    K = ends[-1] + B
    starts_rep = np.zeros(ends[-1] + B, dtype=np.int64)
    ends_rep = np.zeros(ends[-1] + B, dtype=np.int64)

    for i in range(0, N):
        starts_rep[starts[i]-B:ends[i]+B] = starts[i]*np.ones(ends[i]-starts[i]+2*B, np.int64) 
        ends_rep[starts[i]-B:ends[i]+B] = ends[i]*np.ones(ends[i]-starts[i]+2*B, np.int64) 
    
    sim_dat = np.zeros((5, K), dtype=np.float64)
    sim_dat_aux = np.zeros((N,3), dtype=np.float64)
    sim_dat_const = np.zeros((11, K), dtype=np.float64)
    sim_dat_const_aux = np.zeros((N, 3), dtype=np.float64)
    edges = np.zeros((N, 10), dtype=np.int64)
    input_data_temp = []
    vessel_names = []
    junction_functions = []

    nodes = np.zeros((N,3), dtype=np.int64)

    for i in range(0, len(network)):
        end = ends[i]
        start = starts[i]
        M = end-start

        (_edges, _input_data, _sim_dat, 
        _sim_dat_aux, vessel_name, _sim_dat_const,
        _sim_dat_const_aux) = buildVessel(i + 1, network[i], blood, M)

        nodes[i,:] = (int(np.floor(M * 0.25)) - 1, int(np.floor(M * 0.5)) - 1, int(np.floor(M * 0.75)) - 1)

        sim_dat[:,start:end] = _sim_dat
        sim_dat[:,start-B:start:] = _sim_dat[:,0,np.newaxis]*np.ones(B)[np.newaxis,:]
        sim_dat[:,end:end+B] = _sim_dat[:,-1,np.newaxis]*np.ones(B)[np.newaxis,:]
        sim_dat_aux[i,0:2] = _sim_dat_aux
        sim_dat_const[:,start:end] = _sim_dat_const
        sim_dat_const[:,start-B:start:] = _sim_dat_const[:,0,np.newaxis]*np.ones(B)[np.newaxis,:]
        sim_dat_const[:,end:end+B] = _sim_dat_const[:,-1,np.newaxis]*np.ones(B)[np.newaxis,:]
        sim_dat_const_aux[i,:] = _sim_dat_const_aux

        edges[i, :3] = _edges
        input_data_temp.append(_input_data.transpose())

        sim_dat_const[-1,starts[i]-B:ends[i]+B] = sim_dat_const[-1,starts[i]-B:ends[i]+B]/(M)
        vessel_names.append(vessel_name)
    
    input_sizes = [inpd.shape[1] for inpd in input_data_temp]
    input_size = max(input_sizes)
    input_data= np.ones((2*N,input_size), dtype=np.float64)*1000
    for i, inpd in enumerate(input_data_temp):
        input_data[2*i:2*i+2, :inpd.shape[1]] = inpd

    indices = np.arange(0, K, 1)
    indices_1 = indices-starts_rep==-starts_rep[0]+1
    indices_2 = indices-ends_rep==-starts_rep[0]+2

    mask = np.zeros(K, dtype=np.int64)
    mask1 = np.zeros(N, dtype=np.int64)
    
    for j in range(N):
        if sim_dat_const_aux[j,2] == 0: #"none":
            t = edges[j,2]
            edges[j,3] = np.where(edges[:, 1] == t,np.ones_like(edges[:,1]), np.zeros_like(edges[:,1])).sum().astype(int)
            edges[j,6] = np.where(edges[:, 2] == t,np.ones_like(edges[:,2]), np.zeros_like(edges[:,2])).sum().astype(int)
            if edges[j,3] == 2:
                index1 = ends[j]-1
                edges[j,4] = np.where(edges[:, 1] == t)[0][0]
                edges[j,5] = np.where(edges[:, 1] == t)[0][1]
                d1_i_start = starts[edges[j,4]]
                d2_i_start = starts[edges[j,5]]
                sim_dat_const_temp = (sim_dat_const[0,index1],
                                    sim_dat_const[0,d1_i_start],
                                    sim_dat_const[0,d2_i_start],
                                    sim_dat_const[1,index1],
                                    sim_dat_const[1,d1_i_start],
                                    sim_dat_const[1,d2_i_start],
                                    sim_dat_const[2,index1],
                                    sim_dat_const[2,d1_i_start],
                                    sim_dat_const[2,d2_i_start],
                                    sim_dat_const[4, index1],
                                    sim_dat_const[4, d1_i_start],
                                    sim_dat_const[4, d2_i_start])
                junction_functions.append(Partial(solveBifurcationWrapper, sim_dat_const=sim_dat_const_temp,  
                                                  starts=(d1_i_start, d2_i_start), end=index1))
                mask[ends[j]-1:ends[j]+B] = j+1
                mask[starts[edges[j,4]]-B:starts[edges[j,4]]+1] = j + 1
                mask[starts[edges[j,5]]-B:starts[edges[j,5]]+1] = j + 1

            elif edges[j,6] == 1:
                edges[j,7] = np.where(edges[:, 1] == t)[0][0]
                d_i_start = starts[edges[j,7]]
                index1 = ends[j]-1
                sim_dat_const_temp = (sim_dat_const[0,index1],
                                       sim_dat_const[0,d_i_start],
                                       sim_dat_const[1,index1],
                                       sim_dat_const[1,d_i_start],
                                       sim_dat_const[2,index1],
                                       sim_dat_const[2,d_i_start],
                                       sim_dat_const[4, index1],
                                       sim_dat_const[4, d_i_start],
                                       blood.rho)
                junction_functions.append(Partial(solveConjunctionWrapper, sim_dat_const=sim_dat_const_temp,
                                                  start=d_i_start, end=index1))
                mask[ends[j]-1:ends[j]+B] = j+1
                mask[starts[edges[j,7]]-B:starts[edges[j,7]]+1] = j + 1

            elif edges[j,6] == 2:
                temp_1 = np.where(edges[:, 2] == t)[0][0]
                temp_2 = np.where(edges[:, 2] == t)[0][1]
                edges[j,7] = np.minimum(temp_1,temp_2)
                edges[j,8] = np.maximum(temp_1,temp_2)
                edges[j,9] = np.where(edges[:, 1] == t)[0][0]
                p1_i = edges[j,7]
                p2_i = edges[j,8]
                if np.maximum(p1_i, p2_i) != j:
                    junction_functions.append(Partial(lambda dt, sim_dat, sim_dat_aux: (sim_dat, sim_dat_aux)))
                else:
                    index1 = ends[j]
                    d = edges[j,9]
                    p1_i_end = ends[p1_i]
                    d_start = starts[d]
                    sim_dat_const_temp = (sim_dat_const[0,index1],
                                            sim_dat_const[0,p1_i_end-1],
                                            sim_dat_const[0,d_start],
                                            sim_dat_const[1,index1],
                                            sim_dat_const[1,p1_i_end-1],
                                            sim_dat_const[1,d_start],
                                            sim_dat_const[2,index1],
                                            sim_dat_const[2,p1_i_end-1],
                                            sim_dat_const[2,d_start],
                                            sim_dat_const[4,index1],
                                            sim_dat_const[4,p1_i_end-1],
                                            sim_dat_const[4,d_start])
                    junction_functions.append(Partial(solveAnastomosisWrapper, sim_dat_const=sim_dat_const_temp,
                                                  start=d_start, ends=(index1-1,p1_i_end-1)))
                if j == edges[j,8]:
                    mask[starts[edges[j,9]]-B:starts[edges[j,9]]+1] = j + 1
                    mask[ends[edges[j,7]]-1:ends[edges[j,7]]+B] = j + 1
                    mask[ends[edges[j,8]]-1:ends[edges[j,8]]+B] = j + 1
        elif sim_dat_const_aux[j,2] == 1:
            junction_functions.append(Partial(setReflectionOutletBCWrapper, sim_dat_const=sim_dat_const, sim_dat_const_aux=sim_dat_const_aux, 
                                                  edges=edges, starts=starts, rho=blood.rho, B=B+1, ends=ends-1, i=j, index2=ends[j]-2, index3=ends[j]-3))
            mask[ends[j]-1:ends[j]+B] = j+1#j+1
            mask1[j] = j+1#j+1
        else:
            junction_functions.append(Partial(setWindkesselOutletBCWrapper, sim_dat_const=sim_dat_const, sim_dat_const_aux=sim_dat_const_aux, 
                                                  edges=edges, starts=starts, rho=blood.rho, B=B+1, ends=ends-1, i=j, index2=ends[j]-2, index3=ends[j]-3))
            mask[ends[j]-1:ends[j]+B] = j+1#j+1
            mask1[j] = j+1#j+1
        
    mask = np.ones(5, dtype=np.int64)[:,np.newaxis]*mask[np.newaxis,:]
    mask1 = mask1[:,np.newaxis]*np.ones(3, dtype=np.int64)[np.newaxis,:]
    print(mask)
    print(mask1)
    print(junction_functions)
    
        
    #global JUNCTION_FUNCTIONS
    #JUNCTION_FUNCTIONS = junction_functions
    return (sim_dat, sim_dat_aux, sim_dat_const,
            sim_dat_const_aux, N, B,
            edges, input_data, nodes, 
            vessel_names, starts, ends, 
            indices_1, indices_2, junction_functions, mask, mask1)


def buildVessel(ID, vessel_data, blood, M):
    vessel_name = vessel_data["label"]
    s_n = int(vessel_data["sn"])
    t_n = int(vessel_data["tn"])
    L = float(vessel_data["L"])
    E = float(vessel_data["E"])

    R_p, R_d = computeRadii(vessel_data)
    P_ext = getPext(vessel_data)
    dx = L/M
    h_0 = initialiseThickness(vessel_data)
    outlet, Rt, R1, R2, Cc = addOutlet(vessel_data)
    viscT = computeViscousTerm(vessel_data, blood)
    inlet, cardiac_T, input_data = buildHeart(vessel_data)

    Q = np.zeros(M, dtype=np.float64)
    u = np.zeros(M, dtype=np.float64)

    s_pi = np.sqrt(np.pi)
    s_pi_E_over_sigma_squared = s_pi * E / 0.75
    one_over_rho_s_p = 1.0 / (3.0 * blood.rho * s_pi)
    radius_slope = computeRadiusSlope(R_p, R_d, L)

    if h_0 == 0.0:
        R_mean = 0.5 * (R_p + R_d)
        h_0 = computeThickness(R_mean)
    
    R_0 = radius_slope * np.arange(0,M,1) * dx + R_p
    A_0 = np.pi * R_0 * R_0
    A = A_0
    beta = 1/np.sqrt(A_0) * h_0 * s_pi_E_over_sigma_squared
    gamma = beta * one_over_rho_s_p / R_0
    c = waveSpeed(A, gamma)
    wall_E = 3.0 * beta * radius_slope * 1/A_0 * s_pi * blood.rho_inv
    P = pressureSA(np.ones(M,np.float64), beta, P_ext)
    
    if outlet == "wk2":
        R1, R2 = computeWindkesselInletImpedance(R2, blood, A_0, gamma)
        outlet = "wk3"

    W1M0 = u[-1] - 4.0 * c[-1]
    W2M0 = u[-1] + 4.0 * c[-1]

    sim_dat = np.stack((u,Q,A,
                        c,P))
    sim_dat_aux = np.array([W1M0, W2M0])
    sim_dat_const = np.stack((A_0, beta, gamma, 
                              wall_E, P_ext*np.ones(M), viscT*np.ones(M),
                              Rt*np.ones(M), R1*np.ones(M), R2*np.ones(M),
                              Cc*np.ones(M), L*np.ones(M)))
    sim_dat_const_aux = np.array([cardiac_T, int(inlet), int(outlet)])
    edges = np.array([ID, s_n, t_n])
    return(edges, input_data, sim_dat,
           sim_dat_aux, vessel_name, sim_dat_const,
           sim_dat_const_aux)

def computeRadiusSlope(R_p, R_d, L):
    return (R_d - R_p) / L

def computeThickness(R_0_i):
    a = 0.2802
    b = -5.053e2
    c = 0.1324
    d = -0.1114e2
    return R_0_i * (a * np.exp(b * R_0_i) + c * np.exp(d * R_0_i))

def computeRadii(vessel):
    if "R0" not in vessel:
        R_p = float(vessel["Rp"])
        R_d = float(vessel["Rd"])
        return R_p, R_d
    else:
        R_0 = float(vessel["R0"])
        return R_0, R_0

def getPext(vessel):
    if "Pext" not in vessel:
        return 0.0
    else:
        return vessel["Pext"]

def getPhi(vessel):
    if "phi" not in vessel:
        return 0.0
    else:
        return vessel["phi"]

def meshVessel(vessel, L):
    if "M" not in vessel:
        M = max([5, int(np.ceil(L * 1e3))])
    else:
        M = vessel["M"]
        M = max([5, M, int(np.ceil(L * 1e3))])

    return M

def initialiseThickness(vessel):
    if "h0" not in vessel:
        return 0.0
    else:
        return vessel["h0"]

def addOutlet(vessel):
    if "outlet" in vessel:
        outlet = vessel["outlet"]
        if outlet == 3: #"wk3"
            R_t = 0.0
            R_1 = float(vessel["R1"])
            R_2 = float(vessel["R2"])
            C = float(vessel["Cc"])
        elif outlet == 2: #"wk2"
            R_t = 0.0
            R_1 = 0.0
            R_2 = float(vessel["R1"])
            C = float(vessel["Cc"])
        elif outlet == 1: #"reflection"
            R_t = float(vessel["Rt"])
            R_1 = 0.0
            R_2 = 0.0
            C = 0.0
    else: #"none"
        outlet = 0
        R_t = 0.0
        R_1 = 0.0
        R_2 = 0.0
        C = 0.0

    return outlet, R_t, R_1, R_2, C

def computeViscousTerm(vessel_data, blood):
    gamma_profile = vessel_data.get("gamma_profile", 9)
    return 2 * (gamma_profile + 2) * np.pi * blood.mu * blood.rho_inv

def buildHeart(vessel_data):
    if "inlet" in vessel_data:
        input_data = np.loadtxt(vessel_data["inlet file"])
        cardiac_period = input_data[-1, 0]
        return True, cardiac_period, input_data
    else:
        return False, 0.0, np.zeros((1, 2))


def computeWindkesselInletImpedance(R2, blood, A0, gamma):
    R1 = blood.rho * waveSpeed(A0[-1], gamma[-1]) / A0[-1]
    R2 -= R1

    return R1, R2
