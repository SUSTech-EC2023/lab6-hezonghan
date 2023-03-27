
import numpy as np


def objective_function(x):
    return np.sin(10 * np.pi * x) * x + np.cos(2 * np.pi * x) * x
