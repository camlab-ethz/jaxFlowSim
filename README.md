![](param_inference.gif)
# jaxFlowSim

jaxFlowSim is a differentiable 1D-haemodynamics solver. It can infer it's own parameters from data as demonstrated in the animation above. 
This code was written during the course of the thesis "On differentiable simulations of haemodynamic systems" and the corresponding document can be found [here](https://github.com/DiegoRenner/On-fast-simulations-of-cardiac-function).


In order to run the code install the dependencies as described in the dependencies section below. 
To define your own network architecture please refer to the yml structure section below for an example on how to define a network strucutre in yml file.
The repo is split into demos, parsing scripts, plotting scripts, a results folder, the source code, and test scripts. Check the READMEs in the corresponding directories for more on each of these.

## Dependencies


The depencies can be installed by running
```
pip install -r dependencies.txt
```
or 
```
pip install -r dependencies_gpu.txt
```
for installing a gpu enabled JAX.

## Quick Start

After installing the right dependiencies for your setup you can run a simulation by either defining your own network structure as demonstrated in the YML structure below or by using one from the test directory.
Then pass the YML file to the script run_model.py:

```
python run_model.py path/to/YML_file.yml
```
or if you are feeling lucky pass it to the run_model_unsafe.py script for faster but potentially inaccurate results:
```
python run_model_unsafe.py path/to/YML/file.yml
```
.
## YML Structure

```yml
proj_name: <project_name>
blood:
  rho: <blood density [g/L]>
  mu: <blood viscosity [Pa*s]>
solver:
  Ccfl: <courant number>
  conv_tol: <convergence tolerance>
network:
  - label: <name of example input vessel>
    sn: <start node>
    tn: <target node>
    L: <length [m]>
    R0: <reference radius [m]>
    E: <Young's modulus [Pa]>
    inlet: 1
    inlet file: <directory of inlet file>
    inlet number: <inlet number>
  - label: <name of reflection output vessel>
    sn: <start node>
    tn: <target node>
    L: <length [m]>
    R0: <reference radius [m]>
    E: <Young's modulus [Pa]>
    Rt: <reflection coefficient>
  - label: <name of two element windkessel output vessel>
    sn: <start node>
    tn: <target node>
    L: <length [m]>
    R0: <reference radius [m]>
    E: <Young's modulus [Pa]>
    outlet: 2
    R1: <proximal resistance>
    Cc: <compliance [m^3/Pa]>
  - label: <name of three element windkessel output vessel>
    sn: <start node>
    tn: <target node>
    L: <length [m]>
    R0: <reference radius [m]>
    E: <Young's modulus [Pa]>
    outlet: 3
    R1: <proximal resistance>
    R2: <distal resistance>
    Cc: <compliance [m^3/Pa]>
...
```



