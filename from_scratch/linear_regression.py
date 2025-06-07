import numpy as np


class LinearRegressionFS():
    """
    Title: Linear Regression From Scratch

    A simple linear regression algorithm implemented from scratch.

    Attributes
    ----------
    w : np.array, initialized as None
        The weights table.

    Methods
    -------
    fit(X, y)
        Trains the model on provided X and y arrays.
    predict(X)
        Returns predictions made by the model.
    """
    def __init__(self):
        """
        Init weights table.
        """
        self.w = None

    def fit(self, X, y):
        """
        Trains the model on provided X and y arrays.

        Parameters
        ----------
        X : np.ndarray of float
            Table of features.
        y : np.ndarray of float
            Table of targets.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        self.w = np.linalg.pinv(X) @ y

    def predict(self, X):
        """
        Returns predictions made by the model.

        Parameters
        ----------
        X : np.ndarray of float
            Table of features.

        Returns
        -------
        np.ndarray of float
            Predicted targets.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.w
