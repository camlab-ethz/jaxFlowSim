import numpy as np
import matplotlib.pyplot as plt
import sys

openBF_timing_file = sys.argv[1]
jaxFlowSim_timing_file = sys.argv[2]
num_vessels_file = sys.argv[3]
samples = int(sys.argv[4])

openBF_timing = np.loadtxt(openBF_timing_file)
jaxFlowSim_timing = np.loadtxt(jaxFlowSim_timing_file)
num_vessels = np.loadtxt(num_vessels_file)

num_networks = num_vessels.size


openBF_timing_average = np.empty(num_networks)
jaxFlowSim_timing_average = np.empty(num_networks)

for i in range(num_networks):
    openBF_timing_average[i] = np.mean(openBF_timing[i*samples:(i+1)*samples])
    jaxFlowSim_timing_average[i] = np.mean(jaxFlowSim_timing[i*samples:(i+1)*samples])



fig, ax = plt.subplots()
plt.scatter(num_vessels, openBF_timing_average)
#ax.set_ylim(ymin=0)
plt.scatter(num_vessels, jaxFlowSim_timing_average)
#ax.set_ylim(ymin=0)
ax.set_xlabel("#segments")
ax.set_ylabel("t[s]")
plt.title("average compute time over 10 runs")
plt.legend(["openBF", "jaxFlowSim"], loc="upper left")
plt.show()
plt.savefig("timing_benchmark.eps")
