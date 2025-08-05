![](param_inference.gif)

# jaxFlowSim
jaxFlowSim is a differentiable 1D-haemodynamics solver. It can infer it's own parameters from data as demonstrated in the animation above. 
This code was written during the course of the thesis "On differentiable simulations of haemodynamic systems" and the corresponding document can be found [here](https://github.com/DiegoRenner/On-fast-simulations-of-cardiac-function).

In order to run the code install the dependencies as described in the dependencies section below. 
To define your own network architecture please refer to the YML structure section below for an example on how to define a network structure in YML file.
The repo is split into demos, parsing scripts, plotting scripts, a results folder, the source code, and test scripts. Check the READMEs in the corresponding directories for more on each of these.

## Dependencies
In order to install the dependencies first setup a virtual Python environment
```
python -m venv .venv
```
and activate it by running
```
source .venv/bin/activate
```
The dependencies can then be installed by running
```
pip install -r requirements.txt
```
or 
```
pip install -r requirements_gpu.txt
```
for installing a GPU enabled JAX.

## Quick Start
After installing the right dependencies for your setup you can run a simulation by either defining your own network structure as demonstrated in the YML structure below or by using one from the test directory.
If you define your own YML make sure to save it as test/\<modelname\>/\<modelname\>.YML.
Then solve the network as follows:
```
python demos/run_model.py <modelname>
```
or if you are feeling lucky pass it to the run_model_unsafe.py script for faster but potentially inaccurate (no convergence checks) results:
```
python demos/run_model_unsafe.py <modelname>.
```

## YML Structure
```yaml
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
  - label: <name of two element Windkessel output vessel>
    sn: <start node>
    tn: <target node>
    L: <length [m]>
    R0: <reference radius [m]>
    E: <Young's modulus [Pa]>
    outlet: 2
    R1: <proximal resistance>
    Cc: <compliance [m^3/Pa]>
  - label: <name of three element Windkessel output vessel>
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

## Documentation
To generate the docs, load the virtual Python environment in which the dependencies were installed and run 
```
make latexpdf
```
(twice if necessary) or 
```
make html
```
in the docs directory. The files will be generated in docs/_build. 
In order to document new modules to the docs, add a \_\_init\_\_.py file to the module's directory, append the name of the module to the docs/source/modules.rst file, and then run
```
sphinx-apidoc -o source/ path/to/module.
```

## Publication Materials
The Makefile in the publication_materials directory builds the targets: 
- venv
- venv_gpu 
- example_driver
- test
- docs
- clean.

The venv targets will setup a Python virtual environment with or without GPU support. The example driver will plot the loss function and it's derivative of a small bifurcation system with respect to the resistance of an outlet Windkessel model. The test target runs all available models in regular and unsafe (no convergence check) mode while comparing the outputs to precomputed values. A documentation pdf can be generated with the docs target and finally the publication_materials directory can be tidied up with the clean target. Any target can be run from the publication_materials directory with 
```
make <target>
```
