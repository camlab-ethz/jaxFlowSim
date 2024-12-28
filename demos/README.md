## Demos

This contains different scripts that can be used to run the differentiable solver, perform parameter inference and do other miscellaneous tasks. There are the following scripts:

### probe_potential_surface.py

This script plots the derivative and value of the loss function used to perform inference on a resistance parameter of a single bifurcation. The script can be run by simply executing:
```
python probe_potential_surface.py
```

### run_inference.py

Infers the parameters of chosen geometry using optimizers provided by the optax library or the NUTS HMC variant provided by the numpyro library. The script can be run by simply executing:

```
python run_inference.py
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

### steps_opt.py
Optimize the fixed numer of time steps to simulate for when not using convergence checks. This script can be run by executing:
```
python steps_opt.py
```
