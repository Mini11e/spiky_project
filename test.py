import numpy as np
import random



if __name__ == "__main__":

    spikes = np.zeros(3)
    spikes[2] = 1
    mat = np.zeros((3,3))
    mat[0][2] = 100
    mat[0][1] = 300
    t = np.dot(mat, spikes)

        
