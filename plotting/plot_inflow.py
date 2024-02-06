import matplotlib.pyplot as plt
import numpy as np

inflow_files = ["/home/diego/studies/uni/thesis_maths/jaxFlowSim/parsing/models_vm/0007_H_AO_H/flow-files/inflow_1d.flow", 
                "/home/diego/studies/uni/thesis_maths/jaxFlowSim/parsing/models_vm/0029_H_ABAO_H/flow-files/inflow_1d.flow", 
                "/home/diego/studies/uni/thesis_maths/jaxFlowSim/parsing/models_vm/0053_H_CERE_H/flow-files/inflow_1d.flow"]

for inflow_file in inflow_files:
    flow_data = np.loadtxt(inflow_file)
    fig, ax = plt.subplots()
    plt.plot(flow_data[:,0], flow_data[:,1])
    ax.set_xlabel("t[s]")
    ax.set_ylabel("P[?]")
    plt.title(inflow_file)
    plt.show()
    plt.close()

    
