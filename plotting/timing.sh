#!/bin/bash

filenames=("single-artery
                conjunction
                bifurcation
                0007_H_AO_H
                0029_H_ABAO_H                
                0053_H_CERE_H
                adan56")
for filename in $filenames
do
    for i in {1..10}
    do
        python ../main_jax.py $filename >> compute_timing.txt
    done
done