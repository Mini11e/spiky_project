import numpy as np



if __name__ == "__main__":

    dV = np.array([10*i for i in range(5)])
    threshold = 20
    spiked = dV > threshold

    print(dV)
    print(spiked)
    print(dV*spiked)

