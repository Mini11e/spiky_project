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
    
    #first neuron spiked at prev timestep
    spikes = np.array((1,0,0)) # multiply by connectivity matrix, 
    # but make it a connectivity matrix where each is connected with 1,
    # so separate from usual matrix

    prev_spikes = np.dot(connectivity_matrix, spikes) # shows which neuron got input from lateral connection
    # now make a if clause that add all that are 1 to a neuron list placeholder and counts until it encounters a zero, then add to list of neurons
    print(prev_spikes)
    

    connectivity_matrix[0][1]= 2
    print(connectivity_matrix)

    # array of lists, 1 list for each neuron, add element each time if pattern gone through
    # sum over 
        
