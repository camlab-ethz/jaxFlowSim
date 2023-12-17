#jaxFlowSim
This code was written during the course of the thesis "On differentiable simulations of haemodynamic systems" and the corresponding document can be found at github.com/DiegoRenner/On-fast-simulations-of-cardiac-function
It offers the ability to parse models, run models, and infer parameters.
The parsing scripts can be found in the _parsing_ folder.
Models can be parsed from two different formats.
The first format being models provided by _vascularmodel.com_.
In order to parse these kinds of models the files representing a 3D-model from _vascularmodel.com_ needs to be placed in the parsing folder.
Furthermore an encoding of the network structure done by centre-line extraction needs to be placed in the model folder as well.
This network structure can be generated using the _Slicer 3D_ software.
In order to parse a _vascularmodel_ network, once the files have been placed appropriately, run
```
python parse_vm.py <model_name>.
```

The second format that can be parsed is the format used in the openBF-hub repository _github.com/alemelis/openBF-hub_.
In order to parse these models the repository need to be cloned in to the _parsing_ folder.
Then the models can be parsed by running

```
python parse_bf.py <model_name> <model_sub\_dir>}
```

where depending on the model the _model\_sub\_dir_ argument might not be relevant. 
The parsed models are all stored in the _test_ folder.
They can be run by executing the following command in the base directory of the _jaxFlowSim_ repository

```
python run_model.py <model_name>.
```

The results of a successfully simulated model can be found in the folder _results/<model\_name>_.
Finally a parameter inference can be run by executing

```
python run_inference.py <model_name>
```

in the base directory as well.
We note that the inferring of a parameter in this manner is very experimental and requires that the user hand tunes the file _src/inference.py_.
However by running 

```
python run_inference.py 
```

without an argument the toy problem from the last section of the results chapter in _github.com/DiegoRenner/On-fast-simulations-of-cardiac-function_ can be computed.