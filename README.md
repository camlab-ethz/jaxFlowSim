# jaxFlowSim

jaxFlowSim is a differentiable 1D-haemodynamics solver. The repo is split into demos, parsing scripts, plotting scripts, a results folder, the source code, and test scripts. Check below for more on each of these.

The depencies can be installed by running
```
pip install -r dependencies.txt
```
or 
```
pip install -r dependencies_gpu.txt
```
for installing a gpu enabled JAX.

This code was written during the course of the thesis "On differentiable simulations of haemodynamic systems" and the corresponding document can be found [here](https://github.com/DiegoRenner/On-fast-simulations-of-cardiac-function).

## Demos

The subdirectory demos contains different scripts that can be used to run the differentiable solver, perform parameter inference and do other miscellaneous tasks. There are the following scripts:

### compare_inference_optax.py

This script compares output from different parameter inference setups that were prepared using the optax library. The script can be run by simply executing:

```
python compare_inferece_optax.py
```
### probe_potential_surface.py

This script plots the derivative and value of the loss function used to perform inference on a resistance parameter of a single bifurcation. The script can be run by simply executing:
```
python probe_potential_surface.py
```
The bash script probe_potential_surface.sh also allows to run this script in parallel in different ranges of the resistance parameter.

### run_inference_numpyro.py

Infers the resistance parameter of a bifurcation using the NUTS HMC variant provided by the numpyro library. The script can be run by simply executing:

```
python run_inference_nunmpyro.py
```
The bash script run_inference_numpyro.sh also allows to run this script in parallel in different ranges of the resistance parameter.
### run_inference_optax.py

Infers the resistance parameter of a bifurcation using optimizers provided by the optax library. The script can be run by simply executing:

```
python run_inference_optax.py
```
The bash script run_inference_optax.sh also allows to run this script in parallel in different ranges of the resistance parameter:

```
./run_inference_optax.sh <num_pprocesses>
```
### run_model_unsafe.py
Run a model without convergence checks. Ths script can be run by executing:
```
python run_model_unsafe.py <modelname>
```

### run_model.py
Run a model with convergence checks. This script can be run by executing:
```
python run_model.py <modelname>
```

### run_SA_unsafe.py
Run a sensitivity analysis on a the parameters of a bifurcation using simulations without convergence checks. This script can be run by executing:
```
python run_SA_unsafe.py
```

### scaling.py
Make plots of how the runtime scales for different sizes of models. This script can be run by executing:
```
python scaling.py
```

### steps_opt.py
Optimize the fixed numer of time steps to simulate for when not using convergence checks. This script can be run by executing:
```
python steps_opt.py
```

### varied_outlet_params.py
Show how output varies for different output parameters of a bifurcation. This script can be run by executing:
```
python varied_outlet_params.py
```


## Parsing

The parsing scripts can be found in the *parsing* folder. Models can be parsed from two different formats. The first format being models provided by [*vascularmodel.com*](https://vascularmodel.com). In order to parse these kinds of models the files representing a 3D-model from *vascularmodel.com* needs to be placed in the parsing folder. Furthermore an encoding of the network structure done by centre-line extraction needs to be placed in the model folder as well. This network structure can be generated using the *Slicer 3D* software. In order to parse a *vascularmodel* network, once the files have been placed appropriately, run

```
python parse_vm.py <model_name>
```

The second format that can be parsed is the format used in the [openBF-hub repository](https://github.com/alemelis/openBF-hub). In order to parse these models the repository need to be cloned in to the *parsing* folder. Then the models can be parsed by r unning
 
```
python parse_bf.py <model_name> <model_sub_dir>
```

where depending on the model the *model_sub_dir* argument might not be relevant. The parsed models are all stored in the *test* folder.

## Plotting

TODO

## Results

Store outputs from model simulations.

## Source

Source files for the simulation code.

## Test

TODO
