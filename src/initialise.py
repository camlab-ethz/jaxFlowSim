import numpy as np
import jax.numpy as jnp
import yaml
import os.path
import shutil
from src.utils import waveSpeed, pressureSA
from src.components import Blood

CCFL = None
TOTAL_TIME = None
CONV_TOLL = None
EDGES = None
BLOOD = None
JUMP = None
MESH_SIZE = None
NUM_VESSELS = None
INLET_TYPES = None
OUTLET_TYPES = None
INPUT_DATAS = None
CARDIAC_TS = None
DXS = None
GAMMAS = None
S_15_GAMMA = None
WALLES = None
VISCTS = None
A0S = None
PEXTS = None
BETAS = None
RTS = None
R1S = None
R2S = None
CCS = None
NODE2S = None
NODE3S = None
NODE4S = None

def loadSimulationFiles(input_filename):
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
    keys = ["project name", "network", "blood", "solver"]
    for key in keys:
        if key not in data:
            raise ValueError(f"missing section {key} in YAML input file")

    checkSection(data, "blood", ["mu", "rho"])
    checkSection(data, "solver", ["Ccfl", "cycles", "convergence tolerance"])

    if "jump" not in data["solver"]:
        data["solver"]["jump"] = 100


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
    project_name = data["project name"]

    if "results folder" not in data:
        r_folder = project_name + "_results"
    else:
        r_folder = data["results folder"]

    # delete existing folder and results!
    if os.path.isdir(r_folder):
        shutil.rmtree(r_folder)
    os.mkdir(r_folder)

    shutil.copy2(input_filename, r_folder + "/")
    copyInletFiles(data, r_folder)

    os.chdir(r_folder)


def copyInletFiles(data, r_folder):
    for vessel in data["network"]:
        if "inlet file" in vessel:
            shutil.copy2(vessel["inlet file"], r_folder + "/")


def buildBlood(blood_data):
    mu = blood_data["mu"]
    rho = blood_data["rho"]
    rho_inv = 1.0 / rho

    global BLOOD
    BLOOD = Blood(mu, rho, rho_inv)


def buildArterialNetwork(network):
    N = len(network)
    M = 1
    for i in range(0, N):
        L = float(network[i]["L"])
        _M = meshVessel(network[i], L)
        M = _M if _M>M else M
    edges = np.zeros((N, 10), dtype=np.int64)
    outlet_types = np.zeros(N, dtype=np.int64)
    inlet_types = np.zeros(N, dtype=np.int64)
    # make max input_data size non static
    input_datas = np.zeros((2*N,100), dtype=np.float64)
    cardiac_Ts = np.zeros(N, dtype=np.float64)
    Ls = np.zeros(N, dtype=np.float64)
    betas = np.zeros((N,M), dtype=np.float64)
    A0s = np.zeros((N,M), dtype=np.float64)
    Pexts = np.zeros(N, dtype=np.float64)
    gammas = np.zeros((N,M), dtype=np.float64)
    s_15_gammas = np.zeros((N,M), dtype=np.float64)
    viscTs = np.zeros(N, dtype=np.float64)
    wallEs = np.zeros((N,M), dtype=np.float64)
    Rts = np.zeros(N, dtype=np.float64)
    R1s = np.zeros(N, dtype=np.float64)
    R2s = np.zeros(N, dtype=np.float64)
    node2s = np.zeros(N, dtype=np.int32)
    node3s = np.zeros(N, dtype=np.int32)
    node4s = np.zeros(N, dtype=np.int32)
    Ccs = np.zeros(N, dtype=np.float64)
    mesh_sizes = np.zeros(N, dtype=np.int32)
    sim_dat = jnp.zeros((5, N*M), dtype=jnp.float64)
    sim_dat_aux = jnp.zeros((11, N), dtype=jnp.float64)


    

    start = 0
    for i in range(0, len(network)):
        end = (i+1)*M
        (A, Q, u, c, P, 
            W1M0, W2M0, 
            U00A, U00Q, U01A, U01Q, 
            UM1A, UM1Q, UM2A, UM2Q, 
            vessel_name, M, 
            wallVa, wallVb, 
            last_P_name, last_Q_name, 
            last_A_name, last_c_name, 
            last_u_name, 
            out_P_name, out_Q_name, 
            out_A_name, out_c_name, 
            out_u_name, 
            node2, node3, node4, ID, sn, tn, 
            Rt, R1, R2, Cc, outlet, inlet, 
            input_data, cardiac_T, 
            dx, beta, A0, Pext, gamma, s_15_gamma, 
            viscT, wallE, L)= buildVessel(i + 1, network[i], BLOOD, JUMP, M)
        Ls[i] = L
        outlet_types[i] = outlet
        inlet_types[i] = inlet
        input_datas[i:i+2,:input_data.shape[0]] = input_data.transpose()
        cardiac_Ts[i] = cardiac_T
        betas[i,:] = beta
        A0s[i,:] = A0
        Pexts[i] = Pext
        gammas[i,:] = gamma
        s_15_gammas[i,:] = s_15_gamma
        viscTs[i] = viscT
        wallEs[i,:] = wallE
        Rts[i] = Rt
        R1s[i] = R1
        R2s[i] = R2
        Ccs[i] = Cc
        edges[i, 0] = ID
        edges[i, 1] = sn
        edges[i, 2] = tn
        mesh_sizes[i] = M
        sim_dat = sim_dat.at[0,start:end].set(u)
        sim_dat = sim_dat.at[1,start:end].set(Q)
        sim_dat = sim_dat.at[2,start:end].set(A)
        sim_dat = sim_dat.at[3,start:end].set(c)
        sim_dat = sim_dat.at[4,start:end].set(P)
        sim_dat_aux = sim_dat_aux.at[0,i].set(W1M0)
        sim_dat_aux = sim_dat_aux.at[1,i].set(W2M0)
        sim_dat_aux = sim_dat_aux.at[2,i].set(U00Q)
        sim_dat_aux = sim_dat_aux.at[3,i].set(U00A)
        sim_dat_aux = sim_dat_aux.at[4,i].set(U01Q)
        sim_dat_aux = sim_dat_aux.at[5,i].set(U01A)
        sim_dat_aux = sim_dat_aux.at[6,i].set(UM1Q)
        sim_dat_aux = sim_dat_aux.at[7,i].set(UM1A)
        sim_dat_aux = sim_dat_aux.at[8,i].set(UM2Q)
        sim_dat_aux = sim_dat_aux.at[9,i].set(UM2A)
        sim_dat_aux = sim_dat_aux.at[10,i].set(0.0) #Pc
        node2s[i] = node2
        node3s[i] = node3
        node4s[i] = node4
        start = end

    for j in np.arange(0,edges.shape[0],1):
        i = edges[j,0]-1
        if outlet == 0: #"none":
            t = edges[j,2]
            edges[j,3] = jnp.where(edges[:, 1] == t,jnp.ones_like(edges[:,1]), jnp.zeros_like(edges[:,1])).sum().astype(int)
            edges[j,6] = jnp.where(edges[:, 2] == t,jnp.ones_like(edges[:,2]), jnp.zeros_like(edges[:,2])).sum().astype(int)
            if edges[j,0] == 2:
                edges[j,4] = jnp.where(edges[:, 1] == t)[0][0]
                edges[j,5] = jnp.where(edges[:, 1] == t)[0][1]

            elif edges[j,6] == 1:
                edges[j,7] = jnp.where(edges[:, 1] == t)[0][0]

            elif edges[j,6] == 2:
                temp1 = jnp.where(edges[:, 2] == t)[0][0]
                temp2 = jnp.where(edges[:, 2] == t)[0][1]
                edges[j,7] = jnp.minimum(temp1,temp2)#jnp.where(edges[:, 2] == t)[0][0]
                edges[j,8] = jnp.maximum(temp1,temp2)#jnp.where(edges[:, 2] == t)[0][1]
                edges[j,9] = jnp.where(edges[:, 1] == t)[0][0]

    global EDGES
    EDGES = edges

    global MESH_SIZE
    MESH_SIZE=mesh_sizes.max()
    global NUM_VESSELS
    NUM_VESSELS = len(network)
    global OUTLET_TYPES
    OUTLET_TYPES = outlet_types
    global INLET_TYPES
    INLET_TYPES = inlet_types
    global INPUT_DATAS
    INPUT_DATAS = input_datas
    global CARDIAC_TS
    CARDIAC_TS = cardiac_Ts
    global BETAS
    BETAS = betas
    global A0S
    A0S = A0s
    global VISCTS
    VISCTS = viscTs
    global WALLES
    WALLES = wallEs
    global GAMMAS
    GAMMAS = gammas
    global S_15_GAMMA
    S_15_GAMMA = s_15_gammas
    global PEXTS
    PEXTS = Pexts
    global RTS
    RTS = Rts
    global R1S
    R1S = R1s
    global R2S
    R2S = R2s
    global CCS
    CCS = Ccs
    dxs = Ls/MESH_SIZE
    
    
    global DXS
    DXS = dxs
    global NODE2S
    NODE2S = node2s
    global NODE3S
    NODE3S = node3s
    global NODE4S
    NODE4S = node4s

    return sim_dat, sim_dat_aux


def buildVessel(ID, vessel_data, blood, jump, M):
    vessel_name = vessel_data["label"]
    sn = int(vessel_data["sn"])
    tn = int(vessel_data["tn"])
    L = float(vessel_data["L"])
    E = float(vessel_data["E"])

    Rp, Rd = computeRadii(vessel_data)
    Pext = getPext(vessel_data)
    dx = L/M
    h0 = initialiseThickness(vessel_data, M)
    outlet, Rt, R1, R2, Cc = addOutlet(vessel_data)
    viscT = computeViscousTerm(vessel_data, blood)
    inlet, inlet_type, cardiac_T, input_data, inlet_number = buildHeart(vessel_data)
    phi = getPhi(vessel_data)

    Q = jnp.zeros(M, dtype=jnp.float64)
    P = jnp.zeros(M, dtype=jnp.float64)
    A = jnp.zeros(M, dtype=jnp.float64)
    u = jnp.zeros(M, dtype=jnp.float64)
    c = jnp.zeros(M, dtype=jnp.float64)
    A0 = np.zeros(M, dtype=jnp.float64)
    R0 = np.zeros(M, dtype=jnp.float64)
    beta = np.zeros(M, dtype=jnp.float64)
    wallE = np.zeros(M, dtype=jnp.float64)
    gamma = np.zeros(M, dtype=jnp.float64)
    wallVa = np.zeros(M, dtype=jnp.float64)
    wallVb = np.zeros(M, dtype=jnp.float64)
    s_15_gamma = np.zeros(M, dtype=jnp.float64)

    s_pi = np.sqrt(np.pi)
    s_pi_E_over_sigma_squared = s_pi * E / 0.75
    one_over_rho_s_p = 1.0 / (3.0 * blood.rho * s_pi)
    radius_slope = computeRadiusSlope(Rp, Rd, L)

    ah = 0.2802
    bh = -5.053e2
    ch = 0.1324
    dh = -0.1114e2

    if h0 == 0.0:
        Rmean = 0.5 * (Rp + Rd)
        h0 = computeThickness(Rmean, ah, bh, ch, dh)
    
    Cv = 0.5 * s_pi * phi * h0 / (blood.rho * 0.75)

    for i in range(0,M):
        R0[i] = radius_slope * i * dx + Rp
        A0[i] = np.pi * R0[i] * R0[i]
        A = A.at[i].set(A0[i])
        beta[i] = 1/jnp.sqrt(A0)[i] * h0 * s_pi_E_over_sigma_squared
        gamma[i] = beta[i] * one_over_rho_s_p / R0[i]
        s_15_gamma[i] = np.sqrt(1.5 * gamma[i])
        c = c.at[i].set(waveSpeed(A[i], gamma[i]))
        wallE[i] = 3.0 * beta[i] * radius_slope * 1/A0[i] * s_pi * blood.rho_inv
        if phi != 0.0:
            wallVb[i] = Cv * 1/jnp.sqrt(A0)[i] * 1/np.sqrt(dx)
            wallVa[i] = wallVa.at[i].set(0.5 * wallVb[i])
    P = P.at[0:M].set(pressureSA(jnp.ones(M,jnp.float64), beta, Pext))
    


    if outlet == "wk2":
        R1, R2 = computeWindkesselInletImpedance(R2, blood, A0, gamma)
        outlet = "wk3"

    U00A = A0[0]
    U01A = A0[1]
    UM1A = A0[-1]
    UM2A = A0[-2]

    U00Q = 0.0
    U01Q = 0.0
    UM1Q = 0.0
    UM2Q = 0.0

    W1M0 = u[-1] - 4.0 * c[-1]
    W2M0 = u[-1] + 4.0 * c[-1]

    node2 = int(jnp.floor(M * 0.25)) - 1
    node3 = int(jnp.floor(M * 0.5)) - 1
    node4 = int(jnp.floor(M * 0.75)) - 1


    last_A_name = f"{vessel_name}_A.last"
    last_Q_name = f"{vessel_name}_Q.last"
    last_u_name = f"{vessel_name}_u.last"
    last_c_name = f"{vessel_name}_c.last"
    last_P_name = f"{vessel_name}_P.last"

    out_A_name = f"{vessel_name}_A.out"
    out_Q_name = f"{vessel_name}_Q.out"
    out_u_name = f"{vessel_name}_u.out"
    out_c_name = f"{vessel_name}_c.out"
    out_P_name = f"{vessel_name}_P.out"

    return (A, Q, u, c, P, 
            W1M0, W2M0, 
            U00A, U00Q, U01A, U01Q, 
            UM1A, UM1Q, UM2A, UM2Q, 
            vessel_name, M, 
            wallVa, wallVb, 
            last_P_name, last_Q_name, 
            last_A_name, last_c_name, 
            last_u_name, 
            out_P_name, out_Q_name, 
            out_A_name, out_c_name, 
            out_u_name, 
            node2, node3, node4, ID, sn, tn, 
            Rt, R1, R2, Cc, outlet, inlet, 
            input_data, cardiac_T, 
            dx, beta, A0, Pext, gamma, s_15_gamma, 
            viscT, wallE, L)

def computeRadiusSlope(Rp, Rd, L):
    return (Rd - Rp) / L

def computeThickness(R0i, ah, bh, ch, dh):
    return R0i * (ah * jnp.exp(bh * R0i) + ch * jnp.exp(dh * R0i))

def computeRadii(vessel):
    if "R0" not in vessel:
        Rp = float(vessel["Rp"])
        Rd = float(vessel["Rd"])
        return Rp, Rd
    else:
        R0 = float(vessel["R0"])
        return R0, R0

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
        M = max([5, int(jnp.ceil(L * 1e3))])
    else:
        M = vessel["M"]
        M = max([5, M, int(jnp.ceil(L * 1e3))])
    
    #M = 242


    return M

def initialiseThickness(vessel, M):
    if "h0" not in vessel:
        return 0.0
    else:
        return vessel["h0"]

def addOutlet(vessel):
    if "outlet" in vessel:
        outlet = vessel["outlet"]
        if outlet == 3: #"wk3":
            Rt = 0.0
            R1 = float(vessel["R1"])
            R2 = float(vessel["R2"])
            Cc = float(vessel["Cc"])
        elif outlet == 2: #"wk2":
            Rt = 0.0
            R1 = 0.0
            R2 = float(vessel["R1"])
            Cc = float(vessel["Cc"])
        elif outlet == 1: #"reflection":
            Rt = float(vessel["Rt"])
            R1 = 0.0
            R2 = 0.0
            Cc = 0.0
    else:
        outlet = 0 #"none"
        Rt = 0.0
        R1 = 0.0
        R2 = 0.0
        Cc = 0.0

    return outlet, Rt, R1, R2, Cc

def computeViscousTerm(vessel_data, blood):
    gamma_profile = vessel_data.get("gamma_profile", 9)
    return 2 * (gamma_profile + 2) * jnp.pi * blood.mu * blood.rho_inv

def buildHeart(vessel_data):
    if "inlet" in vessel_data:
        inlet_type = vessel_data["inlet"]
        input_data = loadInletData(vessel_data["inlet file"])
        cardiac_period = input_data[-1, 0]
        inlet_number = vessel_data["inlet number"]
        return True, inlet_type, cardiac_period, input_data, inlet_number
    else:
        return False, 0, 0.0, jnp.zeros((1, 2)), 0


def loadInletData(inlet_file):
    numpy_array = np.loadtxt(inlet_file)
    jax_array = jnp.array(numpy_array)
    return jax_array


def computeWindkesselInletImpedance(R2, blood, A0, gamma):
    R1 = blood.rho * waveSpeed(A0[-1], gamma[-1]) / A0[-1]
    R2 -= R1

    return R1, R2

def parseCommandline():
    input_filename = ""
    verbose = False
    out_files = True
    conv_ceil = True

    return input_filename, verbose, out_files, conv_ceil

def buildConst(Ccfl, total_time, conv_toll):
    global CCFL
    CCFL = Ccfl
    global TOTAL_TIME
    TOTAL_TIME = total_time * float(CARDIAC_TS[0])
    global CONV_TOLL
    CONV_TOLL = conv_toll
