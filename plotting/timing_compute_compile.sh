#!/bin/bash

filenames=("single-artery
                conjunction
                bifurcation
                0007_H_AO_H
                0029_H_ABAO_H                
                0053_H_CERE_H
                adan56")

timing_compile_file=timing_compile.txt
timing_compute_file=timing_compute.txt

rm $timing_compile_file
rm $timing_compute_file

for filename in $filenames
do
    for i in {1..10}
    do
        python timing_compile.py $filename >> $timing_compile_file
    done
done

for filename in $filenames
do
    for i in {1..10}
    do
        python timing_compute.py $filename >> $timing_compute_file
    done
done