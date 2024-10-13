#!/bin/bash

openBF_dir="/home/diego/studies/uni/thesis_maths/openBF/test"
jaxFlowSim_dir="/home/diego/studies/uni/thesis_maths/jaxFlowSim"
plotting_dir=$(pwd)
openBF_timing_script=timing_benchmark.jl
jaxFlowSim_timing_script=timing_benchmark.py
plotting_script=plot_benchmark.py
openBF_timing_file=timing_openBF.txt
jaxFlowSim_timing_file=timing_jaxFlowSim.txt
num_vessels_file=num_vessels_benchmark.txt


samples=10

network_names=("single-artery
               conjunction
               bifurcation
               0007_H_AO_H
               0029_H_ABAO_H
               0053_H_CERE_H
               adan56")

cp $openBF_timing_script $openBF_dir
cd $openBF_dir
julia $openBF_timing_script $samples $network_names | grep "Elapsed time =" | awk '{print $4}' > "$plotting_dir/$openBF_timing_file"

rm $openBF_timing_script

cd $plotting_dir
cp $jaxFlowSim_timing_script $jaxFlowSim_dir
cd $jaxFlowSim_dir
source .venv/bin/activate
python $jaxFlowSim_timing_script $samples $num_vessels_file $network_names | grep "elapsed time =" | awk '{print $4}' > "$plotting_dir/$jaxFlowSim_timing_file"
rm $jaxFlowSim_timing_script
cp $num_vessels_file $plotting_dir
rm $num_vessels_file
cd $plotting_dir

python $plotting_script $openBF_timing_file $jaxFlowSim_timing_file $num_vessels_file $samples



