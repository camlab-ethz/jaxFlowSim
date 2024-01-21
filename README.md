# jaxFlowSim

This code was written during the course of the thesis "On differentiable simulations of haemodynamic systems" and the corresponding document can be found [here](https://github.com/DiegoRenner/On-fast-simulations-of-cardiac-function). It offers the ability to parse models, run models, and infer parameters.

## Parsing

The parsing scripts can be found in the *parsing* folder. Models can be parsed from two different formats. The first format being models provided by [*vascularmodel.com*](https://vascularmodel.com). In order to parse these kinds of models the files representing a 3D-model from *vascularmodel.com* needs to be placed in the parsing folder. Furthermore an encoding of the network structure done by centre-line extraction needs to be placed in the model folder as well. This network structure can be generated using the *Slicer 3D* software. In order to parse a *vascularmodel* network, once the files have been placed appropriately, run

```
python parse_vm.py <model_name>
```

The second format that can be parsed is the format used in the [openBF-hub repository](https://github.com/alemelis/openBF-hub). In order to parse these models the repository need to be cloned in to the *parsing* folder. Then the models can be parsed by running

```
python parse_bf.py <model_name> <model_sub_dir>
```

where depending on the model the *model_sub_dir* argument might not be relevant. The parsed models are all stored in the *test* folder.

## Running Models

The parsed models can be run by executing the following command in the base directory of the *jaxFlowSim* repository

```
python run_model.py <model_name>
```

The results of a successfully simulated model can be found in the folder *results/<model_name>*.

## Parameter Inference

Parameter inference can be run by executing

```
python run_inference.py <model_name>
```

in the base directory as well. We note that the inferring of a parameter in this manner is very experimental and requires that the user hand tunes the file *src/inference.py*. However by running

```
python run_inference.py 
```

without an argument the toy problem from the last section of the results chapter in the [thesis](https://github.com/DiegoRenner/On-fast-simulations-of-cardiac-function) can be computed.