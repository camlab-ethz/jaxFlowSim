# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the path to the potential surface file
potential_surface_file = "results/potential_surface/potential_surface_new.txt"

# Change directory to the parent directory of the script file
# Ensure that the script runs relative to the directory structure
os.chdir(os.path.dirname(__file__) + "/..")

# Load the potential surface data from the file
# The file is expected to contain a 2D array with at least three columns
potential_surface = np.loadtxt(potential_surface_file)

# Sort the data based on the first column (R values)
potential_surface = potential_surface[potential_surface[:, 0].argsort()]

# Create a figure and a single set of axes for the first plot
fig, ax1 = plt.subplots()

# Plot the loss function (second column) against R (first column) using red color
color = "tab:red"
ax1.set_xlabel("R")  # Label for the x-axis
ax1.set_ylabel("loss", color=color)  # Label for the y-axis
ax1.plot(
    potential_surface[:, 0], potential_surface[:, 1], color=color
)  # Plot the loss curve
ax1.tick_params(axis="y", labelcolor=color)  # Set the y-axis tick labels color

# Create a second set of axes that shares the same x-axis for gradient plotting
ax2 = ax1.twinx()

# Plot the gradient (third column) against R (first column) using blue color
color = "tab:blue"
ax2.set_ylabel("grad", color=color)  # Label for the second y-axis
ax2.plot(
    potential_surface[:, 0], potential_surface[:, 2], color=color
)  # Plot the gradient curve
ax2.tick_params(axis="y", labelcolor=color)  # Plot the gradient curve

# Adjust the layout to ensure proper spacing and avoid label clipping
fig.tight_layout()  # otherwise the right y-label is slightly clipped

# Adjust the layout to ensure proper spacing and avoid label clipping
plt.show()

# Close the plot to free up memory resources
plt.close()
