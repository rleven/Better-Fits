import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import scipy.stats as stats

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

# * Test data
x = np.linspace(0, 10, 40)
y = [0, 0, 0, -1, -5, -20, -25, -26, -25, -23, -20, -19, -18, -15, -10, -5, 0, 1, 2, 3, 2, 1, 0, -1, -2, -2, -2, -1, 0, 1, 2, 2, 2, 1, 0, -1, -1, -1, 0, 1]
dat = (x, y)

class BFModels:
    def __init__(self):
        """
        Initialize the BFModels (Better Fit Models) class.
        This class is designed to manage fit models, including loading data,
        smart window selection, and retrieving model information.
        """
        self.models = {"Exponentials": self.exponentials_model, "Linear": None, "Polynomial": None}
        self.data = None
        self.fit_results = None
        self.fit_summary = None
    
    def load_data(self, data):
        """
        Load data from a file path, numpy array, pandas DataFrame, or pandas Series.
        If a string is provided, it is treated as a file path.
        If a numpy array, pandas DataFrame, or pandas Series is provided, it is used directly.
        Supports delimiters: comma, tab, and space.
        """
        if isinstance(data, str):
            # Try loading as a text file with different delimiters
            for delim in [',', '\t', ' ']:
                try:
                    self.data = np.loadtxt(data, delimiter=delim)
                    break
                except Exception:
                    self.data = None
            if self.data is None:
                # Try pandas as fallback
                try:
                    self.data = pd.read_csv(data, sep=None, engine='python', header=None).values
                except Exception as e:
                    raise ValueError(f"Failed to load data from file: {e}")
        else:
            self.data = np.asarray(data)

    def gradient(self):
        """
        Calculate the gradient of the loaded data.
        This method should compute the gradient of the y-values with respect to x-values.
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        x = self.data[:, 0]
        y = self.data[:, 1]
        gradient = np.gradient(y, x)
        return gradient

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

    def smart_selec(self):
        """
        Smart window selection based on the loaded data.
        This method should implement logic to select a window for fitting.
        It should analyze the data and determine an appropriate range for fitting.
        Returns a tuple of x and y values for the selected window.
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        else:
            # Example logic for smart selection
            x = self.data[:, 0]
            y = self.data[:, 1]
            grad = self.gradient()
            # Find indices of min and max values of the gradient
            min_grad_index = np.argmin(grad)
            max_grad_index = np.argmax(grad)
            # Check which of the indexes is larger
            indexes = [min_grad_index, max_grad_index]
            start_index = max(indexes)
            end_index = len(x)-1
            return x[start_index:end_index], y[start_index:end_index]
            

    def convoluted_model(self):
        """
        Fit a convoluted model to the data.
        This method should implement the logic for fitting a convoluted model.
        """
        if self.data is None:
            raise ValueError("No data loaded.")
        # Example placeholder for convoluted model fitting
        # Actual implementation would depend on the specific model requirements
        return "Convoluted model fitting not implemented."

    def list_models(self):
        return print(list(self.models.keys()))


modela = BFModels()
modela.load_data("downloads/RawData.txt")
x = modela.data[:, 0]
y = modela.data[:, 1]
plt.plot(x, y, '-', label='Data')
plt.plot(x, modela.gradient(), '--', label='Gradient')
plt.show()
