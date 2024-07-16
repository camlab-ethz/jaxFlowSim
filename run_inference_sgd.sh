#!/bin/bash

source venv/bin/activate

for i in {0..0}
do
	#screen -dmS inference_ensemble$i "source venv/bin/activate; python run_inference6.py bifurcation $i" &
	screen -dmS inference_ensemble_sgd$i bash -c "source venv/bin/activate; python run_inference_sgd.py bifurcation $i" & 
	sleep 10;
	#python run_inference6.py bifurcation $i &
done

