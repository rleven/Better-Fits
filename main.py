import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import scipy.stats as stats

class BFModels:
    def __init__(self):
        """
        Initialize the BFModels (Better Fit Models) class.
        This class is designed to manage fit models, including loading data,
        adding models, and retrieving model information.
        """
        self.models = {}
        self.data = None
        self.fit_results = None
        self.fit_summary = None
    
    def load_data(self, data):
        """
        Load data from a file path, numpy array, pandas DataFrame, or pandas Series.
        If a string is provided, it is treated as a file path.
        If a numpy array, pandas DataFrame, or pandas Series is provided, it is used directly.
        """
        if isinstance(data, str):
            # Try loading as a text file, then as a CSV
            try:
                self.data = np.loadtxt(data)
            except (ValueError, OSError):
                try:
                    self.data = pd.read_csv(data, header=None).squeeze().values
                except Exception as e:
                    raise ValueError(f"Failed to load data from file: {e}")
        else:
            self.data = np.asarray(data)

    def add_model(self, name, model):
        self.models[name] = model

    def exponentials_model(self):
        """
        Fit self.data to a triple exponential function.
        Assumes self.data is a 2D array or DataFrame with columns x and y.
        Returns optimal parameters and the covariance matrix.
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        # Handle both DataFrame and ndarray
        if isinstance(self.data, pd.DataFrame):
            x = self.data.iloc[:, 0].values
            y = self.data.iloc[:, 1].values
        else:
            x = self.data[:, 0]
            y = self.data[:, 1]
        # Provide initial guesses for all 9 parameters
        p0 = [1, -1, 0, 1, -0.1, 0, 1, -0.01, 0]
        popt, pcov = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
        return popt, pcov

    def get_model(self, name):
        return self.models.get(name, None)

    def list_models(self):
        return list(self.models.keys())
    
def exp_func(x, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    """
    Triple exponential function:
    a1 * exp(b1 * x) + c1 + a2 * exp(b2 * x) + c2 + a3 * exp(b3 * x) + c3
    """
    return (
        a1 * np.exp(b1 * x) + c1 +
        a2 * np.exp(b2 * x) + c2 +
        a3 * np.exp(b3 * x) + c3
    )



modela = BFModels()
modela.load_data('downloads/simulated_signal.txt')

