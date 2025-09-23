# non-default GPy Sparse Regressor model definition

from sklearn.preprocessing import StandardScaler
import GPy

def build_gpy_surrogate_model(xtrain, ytrain, use_sparse):
    # Ensure correct shape for ytrain
    if ytrain.ndim == 1:
        ytrain = ytrain.reshape(-1, 1)

    # Normalize inputs and outputs
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    xtrain_scaled = x_scaler.fit_transform(xtrain)
    ytrain_scaled = y_scaler.fit_transform(ytrain)

    input_dim = xtrain.shape[1]

    if use_sparse:
        model = GPy.models.SparseGPRegression(
            xtrain_scaled, ytrain_scaled)
    else:
        model = GPy.models.GPRegression(xtrain_scaled, ytrain_scaled)

    # Prediction function wrapper
    def predict_fn(xtest):
        xtest_scaled = x_scaler.transform(xtest)
        y_scaled_pred, _ = model.predict(xtest_scaled)
        return y_scaler.inverse_transform(y_scaled_pred)

    return model, predict_fn