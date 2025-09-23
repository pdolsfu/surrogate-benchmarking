# python surrogates batch benchmarking on the specified function in line 217

import os
import time
import winsound
import numpy as np
from func_defs import *

import GPy
import xgboost as xgb
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, IDW, KRG, KPLS, KPLSK, RBF as smtRBF, GPX
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import r2_score


def init(f, ndoe, ntest):
    var = []
    # ||| Construction of the DOE |||
    sampling_DOE = LHS(xlimits=f.xlimits, criterion="ese", random_state=1)
    xtrain = sampling_DOE(ndoe)
    ytrain = f(xtrain)

    # ||| Construction of validation points |||
    sampling_validate = LHS(xlimits=f.xlimits, criterion="ese", random_state=2)
    xtest = sampling_validate(ntest)

    return {'xtrain': xtrain, 'ytrain': ytrain, 'xtest': xtest}

def build_gpy_surrogate_model(xtrain, ytrain, use_sparse=False, num_inducing=20, num_restarts=5, verbose=False):
    # Ensure correct shape for ytrain
    if ytrain.ndim == 1:
        ytrain = ytrain.reshape(-1, 1)

    # Normalize inputs and outputs
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    xtrain_scaled = x_scaler.fit_transform(xtrain)
    ytrain_scaled = y_scaler.fit_transform(ytrain)

    input_dim = xtrain.shape[1]
    kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)

    if use_sparse:
        model = GPy.models.SparseGPRegression(
            xtrain_scaled, ytrain_scaled, kernel)
        model.inducing_inputs = xtrain_scaled[:num_inducing].copy()
    else:
        model = GPy.models.GPRegression(xtrain_scaled, ytrain_scaled, kernel)

    # Optional: Constrain noise variance to be positive and not too small
    model.Gaussian_noise.variance.constrain_bounded(1e-6, 1.0)

    # Optimize with restarts
    model.optimize_restarts(num_restarts=num_restarts, verbose=verbose)

    # Prediction function wrapper
    def predict_fn(xtest):
        xtest_scaled = x_scaler.transform(xtest)
        y_scaled_pred, _ = model.predict(xtest_scaled)
        return y_scaler.inverse_transform(y_scaled_pred)

    return model, predict_fn

def compute_accuracy(y_true, y_surrogate):
    metrics = []
    r2 = r2_score(y_true, y_surrogate)

    mae = np.mean(np.abs(y_true - y_surrogate))
    std_y = np.std(y_true)
    raae = mae / std_y

    max_ae = np.max(np.abs(y_true - y_surrogate))
    rmae = max_ae / std_y

    return {'RAAE': raae, 'RMAE': rmae, 'R2': r2}

def write_accuracy(metric, model_type, ndim, ndoe, time, file_path):
    """
    Append one row to a tab-delimited .dat file with columns:
      time    R2    RAAE    RMAE    model_type
    All floats are in scientific notation with 3 decimal places.
    """
    # Check if we need to write the header
    write_header = not os.path.exists(file_path) or os.stat(file_path).st_size == 0

    with open(file_path, 'a', newline='') as f:
        if write_header:
            f.write("time\tR2\tRAAE\tRMAE\tmodel_type\n")
        # now write one row
        f.write(
            f"{time:.3e}\t"
            f"{metric['R2']:.3e}\t"
            f"{metric['RAAE']:.3e}\t"
            f"{metric['RMAE']:.3e}\t"
            f"{model_type}\n"
        )

def make_pred(model_type, ndim, ndoe, xtest, xtrain, ytrain, f):
    if model_type == 'LS':  # LINEAR
        model = LS(print_prediction=False)
        model.set_training_values(xtrain, ytrain)
        start = time.time()
        model.train()
        end = time.time()
        y_pred = model.predict_values(xtest).ravel()
    elif model_type == 'QP':  # QUADRATIC
        model = QP(print_prediction=False)
        model.set_training_values(xtrain, ytrain)
        start = time.time()
        model.train()
        end = time.time()
        y_pred = model.predict_values(xtest).ravel()
    elif model_type == 'KRG':  # KRIGING
        model = KRG(theta0=[1e-2] * ndim, print_prediction=False)
        model.set_training_values(xtrain, ytrain)
        start = time.time()
        model.train()
        end = time.time()
        y_pred = model.predict_values(xtest).ravel()
    elif model_type == 'KPLS':  # KPLS (Kriging, with Partial Least Squares)
        model = KPLS(n_comp=1, theta0=1 *
                     [1e-2], print_prediction=False, corr="abs_exp")
        model.set_training_values(xtrain, ytrain)
        start = time.time()
        model.train()
        end = time.time()
        y_pred = model.predict_values(xtest).ravel()
    elif model_type == 'KPLSK':  # KPLSK (KPLS, with added steps)
        model = KPLSK(n_comp=2, theta0=[1e-2, 1e-2], print_prediction=False)
        model.set_training_values(xtrain, ytrain)
        start = time.time()
        model.train()
        end = time.time()
        y_pred = model.predict_values(xtest).ravel()
    elif model_type == 'GPX':  # GPX
        model = GPX(theta0=[1e-2])
        model.set_training_values(xtrain, ytrain)
        start = time.time()
        model.train()
        end = time.time()
        y_pred = model.predict_values(xtest).ravel()
    elif model_type == 'RBF':  # RBF
        model = smtRBF(d0=100)
        model.set_training_values(xtrain, ytrain)
        start = time.time()
        model.train()
        end = time.time()
        y_pred = model.predict_values(xtest).ravel()
    elif model_type == 'IDW':  # IDW
        model = IDW(p=2)
        model.set_training_values(xtrain, ytrain)
        start = time.time()
        model.train()
        end = time.time()
        y_pred = model.predict_values(xtest).ravel()
    elif model_type == 'GPy':
        start = time.time()
        model, predict = build_gpy_surrogate_model(
            xtrain, ytrain, use_sparse=False, num_inducing=20, num_restarts=5, verbose=False)
        end = time.time()
        y_pred = predict(xtest)
    elif model_type == 'scikit':
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(1.0, length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, alpha=1e-10, normalize_y=True)
        start = time.time()
        gp.fit(xtrain, ytrain)
        end = time.time()
        y_pred, sigma = gp.predict(xtest, return_std=True)
    elif model_type == 'XGBoost':
        # Ensure ytrain is a 1D array for regression
        ytrain = ytrain.ravel()

        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(xtrain, label=ytrain)
        dtest = xgb.DMatrix(xtest)

        # Define XGBoost regression parameters
        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.3,
            "verbosity": 0
        }

        # Train the model
        start = time.time()
        model = xgb.train(params, dtrain)
        end = time.time()

        # Predict
        y_pred = model.predict(dtest)
    elif model_type == 'MVRSM':
        n_relus = 100
        seed_train = 1
        seed_test = 1
        d = f.xlimits.shape[0]

        # Generate fixed ReLU weights and biases
        rng = np.random.default_rng(seed_train)
        W = rng.standard_normal((n_relus, d))
        b = rng.standard_normal(n_relus)

        # Compute training feature matrix
        Z = np.maximum(0, W @ xtrain.T + b[:, np.newaxis])  # Shape: (n_relus, ndoe)

        # Solve for coefficients c
        start = time.time()
        c = np.linalg.lstsq(Z.T, ytrain, rcond=None)[0]
        end = time.time()
        # Compute test feature matrix and predict
        Z_test = np.maximum(0, W @ xtest.T + b[:, np.newaxis])
        y_pred = (c @ Z_test).ravel()

        # Return test points, predictions, and model info
        model = {"W": W, "b": b, "c": c}


    return y_pred, (end - start)

def main(ndim=0000, ndoe=0000, function_number=0000, ntest=0000):                                                     # modify this line to specify which function

    # Initialize absolute file path for results document
    base_dir = r"\...enter_your_directory...\python_results"                                                          # change file path to match users directory
    folder_name = f"results{function_number}.dat"
    abs_path = os.path.join(base_dir, folder_name)    
    open(abs_path, "w").close()

    # Initialize variables and vectors
    func_name = f"true_f{function_number}"
    f_class  = globals()[func_name]
    f = f_class(ndim=ndim)  
    
    variables = init(f, ndoe, ntest)
    xtrain = variables['xtrain']
    ytrain = variables['ytrain']
    xtest = variables['xtest']

    model_types = ["LS", "QP", "KRG", "KPLS", "KPLSK", "GPX", "RBF", "IDW", "GPy", "scikit"]

    # Pick surrogate model and make predictions
    for model_type in model_types:
        try:
            y_pred, time = make_pred(model_type, ndim, ndoe, xtest, xtrain, ytrain, f)
            y_true = f(xtest)
            metrics = compute_accuracy(y_true, y_pred)
            write_accuracy(metrics, model_type, ndim, ndoe, time, abs_path)
        except Exception as e:
            print(f"⚠️ Error with model_type '{model_type}': {e}")

if __name__ == "__main__":
    main()