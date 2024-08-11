#!/bin/bash

# Use this script to launch probe_potential_surface.py on multiple cores at once, while splitting the range to be probed over all cores.

# Activate the virtual environment
source venv/bin/activate

# Calculate the number of processes to run
num_proc=$(($1-1))

# Loop through the number of processes
for i in $(seq 0 $num_proc)
do
    # Print the current process number
    echo "Starting process $i"

	# Start a new detached screen session for each process
    # The session is named 'probe_potential_surface' followed by the process number
    # Run the Python script 'probe_potential_surface.py' with arguments 'bifurcation', the process number, and the total number of processes
	screen -dmS probe_potential_surface"$i" bash -c "source venv/bin/activate; python probe_potential_surface.py bifurcation $i $1" & 

    # Sleep for 10 seconds to prevent race conditions or resource contention
	sleep 10;
done

echo "All processes started."