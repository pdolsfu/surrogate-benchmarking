### PYTHON CODE

# Run this terminal command to install the necessary Python libraries: pip install numpy GPy smt scikit-learn smt["gpx"]

# With the help of Chat.GPT to fix syntax and implement models, I wrote a Python script that sequentially trains surrogate models on the specified function, compares them to the values of the specified function, and outputs them to a tabular data file. The code is fully modularized so that the process of training all the models on a given function and outputing their results can be done with changing one line of code. This is done by indicating the function number, number of training and test samples, and dimensionality of the function. The general process of the Python code is to generate LHS samples that are fed into the true function. These inputs and outputs are used to train the surrogate model, which is then used to make a list of predictions. The true function, defined in func_defs.py, takes these inputs for those predictions and calculates the true values for the objective function. Error metrics are then calculated and written onto function_xx.

# This is the full list of methods used: LS, QP, KRG, KRG, KPLS, KPLSK, GPX, RBF, IDW, GPy's Spare Regression GP model, and skikit learn's GaussianProcessRegressor. I also attempted to implement XGBoost and MVRSM, but using my specifications, they were not very effective, so further testing was discontinued.

# Function definitions are located in func_defs.py and the script that automates the testing process for each function for all the models is found in surrogate_batch_main.py. The results are found in the python_benchmark_results folder.

# The implementation of the SMT code was straight forward, and its documentation can be found here: https://smt.readthedocs.io. The only hyperparameter values that were modified were

# - Setting RBF theta value = 100

# - KPLSK n_comp value to 2

# The GPy and sklearn surrogate model implementations both utilized a Gaussian Process model with RBF Kernels with non-default settings. Here are the changes made for GPy, the code for the default function definition can be found in "default_GPy_method.py":

# - num_inducing was set to 20 instead of 10, kernel used ARD = True, restarts were not optimized, and noise was not constrained.

# Here are the changes made for the scikit-learn model:

# - n_restarts_optimizer was set to 10 instead of 0, normalize_y set to True

# - the kernel was specified with constant_value_bounds = (1e-3, 1e3) and length_scale_bounds = (1e-2, 1e2)

# Their documentation can be found below, and the results from the models with default hyperparameters (very poor results) are listed in "default_scikit_results.dat" and "default_GPy_results.dat".

# - https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

# - https://gpy.readthedocs.io/en/deploy/
