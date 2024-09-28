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
