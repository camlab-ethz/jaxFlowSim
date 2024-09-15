## Demos

This contains different scripts that can be used to run the differentiable solver, perform parameter inference and do other miscellaneous tasks. There are the following scripts:

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
