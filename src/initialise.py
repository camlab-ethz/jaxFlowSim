import numpy as np
import yaml
import os.path
import shutil
from src.utils import waveSpeed, pressureSA
from src.components import Blood

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
        r_folder = "results/" +  project_name + "_results"
    else:
        r_folder = data["results folder"]

    # delete existing folder and results!
    if os.path.isdir(r_folder):
        shutil.rmtree(r_folder)
    os.makedirs(r_folder, mode = 0o777)

    shutil.copy2(input_filename, r_folder + "/")
    copyInletFiles(data, r_folder)

    #os.chdir(r_folder)


def copyInletFiles(data, r_folder):
    for vessel in data["network"]:
        if "inlet file" in vessel:
            shutil.copy2(vessel["inlet file"], r_folder + "/")


def buildBlood(blood_data):
    mu = blood_data["mu"]
    rho = blood_data["rho"]
    rho_inv = 1.0 / rho

    return Blood(mu, rho, rho_inv)


def buildArterialNetwork(network, J, blood):
    
    B = 2
    N = len(network)
    M0 = meshVessel(network[0], float(network[0]["L"]))
    starts = np.zeros(N, dtype=np.int64)
    ends = np.zeros(N, dtype=np.int64)

    starts[0] = B
    ends[0] = M0 + B
    #ends[0] = 40 + B

    for i in range(1, N):
        L = float(network[i]["L"])
        _M = meshVessel(network[i], L)
        starts[i] = ends[i-1] + 2*B
        ends[i] = starts[i] + _M
        #ends[i] = starts[i] + 40
        #M = _M if _M>M else M

    #M = 40
    #print(M)

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
    # make max input_data size non static
    input_data_temp = []
    vessel_names = []


    nodes = np.zeros((N,3), dtype=np.int64)
    

    for i in range(0, len(network)):
        end = ends[i]
        start = starts[i]
        M = end-start
        (_edges,
        _input_data,
        _sim_dat, 
        _sim_dat_aux, 
        vessel_name,
        #wallVa, wallVb, 
        #last_P_name, last_Q_name, 
        #last_A_name, last_c_name, 
        #last_u_name, 
        #out_P_name, out_Q_name, 
        #out_A_name, out_c_name, 
        #out_u_name, 
        _sim_dat_const,
        _sim_dat_const_aux)= buildVessel(i + 1, network[i], blood, J, M)

        nodes[i,:] = (int(np.floor(M * 0.25)) - 1, int(np.floor(M * 0.5)) - 1, int(np.floor(M * 0.75)) - 1)

        sim_dat[:,start:end] = _sim_dat
        sim_dat[:,start-B:start:] = _sim_dat[:,0,np.newaxis]*np.ones(B)[np.newaxis,:]
        sim_dat[:,end:end+B] = _sim_dat[:,-1,np.newaxis]*np.ones(B)[np.newaxis,:]
        sim_dat_aux[i,0:2] = _sim_dat_aux
        sim_dat_const[:,start:end] = _sim_dat_const
        sim_dat_const[:,start-B:start:] = _sim_dat_const[:,0,np.newaxis]*np.ones(B)[np.newaxis,:]
        sim_dat_const[:,end:end+B] = _sim_dat_const[:,-1,np.newaxis]*np.ones(B)[np.newaxis,:]
        sim_dat_const_aux[i,:] = _sim_dat_const_aux
        #sim_dat_const_aux = sim_dat_const_aux.at[:,i].set(_sim_dat_const_aux)

        edges[i, :3] = _edges
        #input_data[2*i:2*i+2,:_input_data.shape[0]] = _input_data.transpose()
        input_data_temp.append(_input_data.transpose())
        #input_data[2*i+1,:_input_data.shape[0]] = -input_data[2*i+1,:_input_data.shape[0]]/(1e4)

        sim_dat_const[-1,starts[i]-B:ends[i]+B] = sim_dat_const[-1,starts[i]-B:ends[i]+B]/(M)
        vessel_names.append(vessel_name)
    
    input_sizes = [inpd.shape[1] for inpd in input_data_temp]
    input_size = max(input_sizes)
    input_data= np.ones((2*N,input_size), dtype=np.float64)*1000
    for i, inpd in enumerate(input_data_temp):
        input_data[2*i:2*i+2, :inpd.shape[1]] = inpd

    indices = np.arange(0, K, 1)
    indices1 = indices-starts_rep==-starts_rep[0]+1
    indices2 = indices-ends_rep==-starts_rep[0]+2
    

    #sim_dat_const[-1,:] = sim_dat_const[-1,:]/M
    #sim_dat_const_aux = sim_dat_const_aux.at[0,:].set(sim_dat_const_aux[0,:]/M)

    #jnp.set_printoptions(threshold=sys.maxsize)
    #print(edges)
    for j in np.arange(0,edges.shape[0],1):
        #print(edges[j,:])
        i = edges[j,0]-1
        if sim_dat_const_aux[i,2] == 0: #"none":
            t = edges[j,2]
            edges[j,3] = np.where(edges[:, 1] == t,np.ones_like(edges[:,1]), np.zeros_like(edges[:,1])).sum().astype(int)
            edges[j,6] = np.where(edges[:, 2] == t,np.ones_like(edges[:,2]), np.zeros_like(edges[:,2])).sum().astype(int)
            if edges[j,3] == 2:
                edges[j,4] = np.where(edges[:, 1] == t)[0][0]
                edges[j,5] = np.where(edges[:, 1] == t)[0][1]

            elif edges[j,6] == 1:
                edges[j,7] = np.where(edges[:, 1] == t)[0][0]

            elif edges[j,6] == 2:
                #try:
                temp1 = np.where(edges[:, 2] == t)[0][0]
                temp2 = np.where(edges[:, 2] == t)[0][1]
                edges[j,7] = np.minimum(temp1,temp2)#jnp.where(edges[:, 2] == t)[0][0]
                edges[j,8] = np.maximum(temp1,temp2)#jnp.where(edges[:, 2] == t)[0][1]
                edges[j,9] = np.where(edges[:, 1] == t)[0][0]
                #except IndexError:
                #    edges[j,6] = 1
                #    temp1 = jnp.where(edges[:, 2] == t)[0][0]
                #    temp2 = jnp.where(edges[:, 2] == t)[0][1]
                #    edges[j,7] = temp1 if temp1 != j else temp2




    return sim_dat, sim_dat_aux, sim_dat_const, sim_dat_const_aux, N, B, edges, input_data, nodes, vessel_names, starts, ends, indices1, indices2


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
    #phi = getPhi(vessel_data)

    Q = np.zeros(M, dtype=np.float64)
    u = np.zeros(M, dtype=np.float64)
    #wallVa = np.zeros(M, dtype=np.float64)
    #wallVb = np.zeros(M, dtype=np.float64)

    s_pi = np.sqrt(np.pi)
    s_pi_E_over_sigma_squared = s_pi * E / 0.75
    one_over_rho_s_p = 1.0 / (3.0 * blood.rho * s_pi)
    radius_slope = computeRadiusSlope(Rp, Rd, L)


    if h0 == 0.0:
        Rmean = 0.5 * (Rp + Rd)
        h0 = computeThickness(Rmean)
    
    #Cv = 0.5 * s_pi * phi * h0 / (blood.rho * 0.75)

    R0 = radius_slope * np.arange(0,M,1) * dx + Rp
    A0 = np.pi * R0 * R0
    A = A0
    beta = 1/np.sqrt(A0) * h0 * s_pi_E_over_sigma_squared
    gamma = beta * one_over_rho_s_p / R0
    c = waveSpeed(A, gamma)
    wallE = 3.0 * beta * radius_slope * 1/A0 * s_pi * blood.rho_inv
    #if phi != 0.0:
    #    wallVb = Cv * 1/jnp.sqrt(A0) * 1/np.sqrt(dx)
    #    wallVa = 0.5 * wallVb
    P = pressureSA(np.ones(M,np.float64), beta, Pext)
    


    if outlet == "wk2":
        R1, R2 = computeWindkesselInletImpedance(R2, blood, A0, gamma)
        outlet = "wk3"

    #U00A = A0[0]
    #U01A = A0[1]
    #UM1A = A0[-1]
    #UM2A = A0[-2]

    #U00Q = 0.0
    #U01Q = 0.0
    #UM1Q = 0.0
    #UM2Q = 0.0

    W1M0 = u[-1] - 4.0 * c[-1]
    W2M0 = u[-1] + 4.0 * c[-1]



    #last_A_name = f"{vessel_name}_A.last"
    #last_Q_name = f"{vessel_name}_Q.last"
    #last_u_name = f"{vessel_name}_u.last"
    #last_c_name = f"{vessel_name}_c.last"
    #last_P_name = f"{vessel_name}_P.last"

    #out_A_name = f"{vessel_name}_A.out"
    #out_Q_name = f"{vessel_name}_Q.out"
    #out_u_name = f"{vessel_name}_u.out"
    #out_c_name = f"{vessel_name}_c.out"
    #out_P_name = f"{vessel_name}_P.out"

    sim_dat = np.stack((u,Q,A,c,P))
    sim_dat_aux = np.array([W1M0, W2M0])
    #        U00Q, U00A, U01Q, U01A, 
    #        UM1Q, UM1A, UM2Q, UM2A])
    sim_dat_const = np.stack((A0, beta, gamma, wallE, 
                              Pext*np.ones(M),
                              viscT*np.ones(M),
                              Rt*np.ones(M),
                              R1*np.ones(M),
                              R2*np.ones(M),
                              Cc*np.ones(M),
                              L*np.ones(M)))
    sim_dat_const_aux = np.array([cardiac_T, inlet, outlet])
    edges = np.array([ID, sn, tn])
    return(edges,
           input_data,
           sim_dat,
           sim_dat_aux,
           vessel_name, 
           #wallVa, wallVb, 
           #last_P_name, last_Q_name, 
           #last_A_name, last_c_name, 
           #last_u_name, 
           #out_P_name, out_Q_name, 
           #out_A_name, out_c_name, 
           #out_u_name, 
           sim_dat_const,
           sim_dat_const_aux)

def computeRadiusSlope(Rp, Rd, L):
    return (Rd - Rp) / L

def computeThickness(R0i):
    ah = 0.2802
    bh = -5.053e2
    ch = 0.1324
    dh = -0.1114e2
    return R0i * (ah * np.exp(bh * R0i) + ch * np.exp(dh * R0i))

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
        M = max([5, int(np.ceil(L * 1e3))])
    else:
        M = vessel["M"]
        M = max([5, M, int(np.ceil(L * 1e3))])
    
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
    return 2 * (gamma_profile + 2) * np.pi * blood.mu * blood.rho_inv

def buildHeart(vessel_data):
    if "inlet" in vessel_data:
        inlet_type = vessel_data["inlet"]
        input_data = np.loadtxt(vessel_data["inlet file"])
        cardiac_period = input_data[-1, 0]
        inlet_number = vessel_data["inlet number"]
        return True, inlet_type, cardiac_period, input_data, inlet_number
    else:
        return False, 0, 0.0, np.zeros((1, 2)), 0


def computeWindkesselInletImpedance(R2, blood, A0, gamma):
    R1 = blood.rho * waveSpeed(A0[-1], gamma[-1]) / A0[-1]
    R2 -= R1

    return R1, R2
