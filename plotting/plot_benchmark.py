import numpy as np
import matplotlib.pyplot as plt
import sys


def main(openBF_timing_file, jaxFlowSim_timing_file, num_vessels_file, samples):
    """
    Compare the average compute times of openBF and jaxFlowSim simulations
    over multiple runs and visualize the results.

    Args:
        openbf_file (str): Path to the openBF timing data file.
        jaxflowsim_file (str): Path to the jaxFlowSim timing data file.
        num_vessels_file (str): Path to the file containing the number of vessels.
        samples (int): Number of sample runs for each simulation.
    """

    # Load timing and network size data
    openBF_timing = np.loadtxt(openBF_timing_file)  # openBF timing data
    jaxFlowSim_timing = np.loadtxt(jaxFlowSim_timing_file)  # jaxFlowSim timing data
    num_vessels = np.loadtxt(
        num_vessels_file
    )  # Number of vessel segments in each network

    # Determine the number of networks based on the num_vessels array size
    num_networks = num_vessels.size

    # Initialize arrays to store average timings for openBF and jaxFlowSim
    openBF_timing_average = np.empty(num_networks)
    jaxFlowSim_timing_average = np.empty(num_networks)

    # Compute the average timing over the sample runs for each network
    for i in range(num_networks):
        openBF_timing_average[i] = np.mean(
            openBF_timing[i * samples : (i + 1) * samples]
        )
        jaxFlowSim_timing_average[i] = np.mean(
            jaxFlowSim_timing[i * samples : (i + 1) * samples]
        )

    # Plot the results
    _, ax = plt.subplots()

    # Scatter plot for openBF and jaxFlowSim timings
    plt.scatter(num_vessels, openBF_timing_average)
    plt.scatter(num_vessels, jaxFlowSim_timing_average)

    # Set axis labels and title
    ax.set_xlabel("#segments")
    ax.set_ylabel("t[s]")
    plt.title("average compute time over 10 runs")

    # Add legend and save the plot
    plt.legend(["openBF", "jaxFlowSim"], loc="upper left")
    plt.savefig("timing_benchmark.eps")
    plt.show()


# Run the script by calling main with command line arguments
if __name__ == "__main__":
    # Parse command-line arguments
    openBF_timing_file = sys.argv[1]
    jaxFlowSim_timing_file = sys.argv[2]
    num_vessels_file = sys.argv[3]
    samples = int(sys.argv[4])

    # Call the main function
    main(openBF_timing_file, jaxFlowSim_timing_file, num_vessels_file, samples)
