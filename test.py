import numpy as np



if __name__ == "__main__":
    
    resting_potential = 0
    threshold = 10
    V = np.zeroes()
    fired = V > threshold
    V[fired]    = resting_potential
    print(V)
    print(V[fired])