import numpy as np


class LossAndDerivatives:
    @staticmethod
    def mse(X, y, w):
        return np.mean((X.dot(w) - y) ** 2)

    @staticmethod
    def mae(X, y, w):
        return np.mean(abs(X.dot(w) - y))

    @staticmethod
    def l2_reg(w):
        return np.sum(np.square(w))

    def l1_reg(w):
        return np.linalg.norm(w, ord=1)

    def no_reg(w):
        return 0

    def mse_derivative(X, y, w):
        return (X.T.dot(X.dot(w) - y)) * 2 / y.shape[0]

    def mae_derivative(X, Y, w):
        n_observations, n_features = X.shape
        target_dimentionality = Y.shape[1] if len(Y.shape) > 1 else 1

        prediction = np.dot(X, w)

        error = np.abs(prediction - Y)

        if target_dimentionality > 1:
            derivative = np.dot(X.T, np.sign(prediction - Y)) / n_observations
        else:
            derivative = np.dot(X.T, np.sign(prediction - Y)) / n_observations

        return derivative

    def l2_reg_derivative(w):
        return 2 * w

    def l1_reg_derivative(w):
        return np.sign(w)