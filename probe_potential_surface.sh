#!/bin/bash

source venv/bin/activate

num_proc=$(($1-1))
for i in $(seq 0 $num_proc)
do
	echo $i
	#screen -dmS inference_ensemble$i "source venv/bin/activate; python run_inference6.py bifurcation $i" &
	screen -dmS probe_potential_surface$i bash -c "source venv/bin/activate; python probe_potential_surface.py bifurcation $i $1" & 
	sleep 10;
	#python run_inference6.py bifurcation $i &
done

