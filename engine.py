import numpy as np

def differentiate(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Calculate the differential value for every step
    :param y: function values according to given x
    :param x: variable values
    :return: differential value of each step
    """
    dx = x[1] - x[0]
    dydx= np.gradient(y, dx)
    return dydx