import numpy as np
import random
import MAIN
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline, BSpline


if __name__ == "__main__":


    # 3 neurons
    # connectivity matrix: 0->1, 1->2, 2->0

    connectivity_matrix = np.array(((0,0,1),
                                   (1,0,0),
                                   (0,1,0)))
    

    connectivity_matrix[0][1]= 2
    print(connectivity_matrix)
        
