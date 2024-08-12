#!/bin/bash
# This script starts multiple detached screen sessions to run the 'run_inference_numpyro.py' script
# for an ensemble training of inferences based on the NUTS optimizer provided by numpyro.
# Each process is started with a different initial value according to the num_process variable.

# Activate the virtual environment
source venv/bin/activate

# Number of processes to run
num_processes=8

# Loop through the specified number of processes
for i in $(seq 1 $num_processes)
do
    # Print the current process number
    echo "Starting process $i"

	# Start a new detached screen session for each process
    # The session is named 'inference_ensemble_numpyro' followed by the process number
    # Run the Python script 'run_inference_numpyro.py' with arguments 'bifurcation' and the process number
	screen -dmS inference_ensemble_numpyro"$i" bash -c "source .venv/bin/activate; python run_inference_nupyro.py $i" & 

	# Sleep for 10 seconds to prevent race conditions or resource contention
	sleep 10;
done

echo "All processes started."

