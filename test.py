import numpy as np
import random
import MAIN
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline, BSpline


if __name__ == "__main__":

    locs = np.round(np.linspace(0.4, 0.6, 3), 2) #np.round(np.linspace(0.8, 1.5, 5), 2)
    scales = np.round(np.linspace(0.6, 1.2, 6), 2)
    

    for loc in locs:
        for scale in scales:
            noise = np.random.gamma(shape=loc, scale=scale)
            print(noise)
        
