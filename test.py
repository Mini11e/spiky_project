import numpy as np
import random



if __name__ == "__main__":

    spikes = np.zeros(3)
    spikes[2] = 1
    mat = np.zeros((3,3))
    mat[0][2] = 100
    mat[0][1] = 300
    t = np.dot(mat, spikes)

    percentage = 0.13
    neurons = 20

    
    distribute_func = lambda m, n: (lambda base, remainder: [base + (1 if i < remainder else 0)for i in range(n)])(m // n,m % n)
    connections_per_neuron = distribute_func((round(neurons*percentage*neurons)), neurons)
    print("ooooo")
    print(connections_per_neuron)
    np.random.seed(30)
    np.random.shuffle(connections_per_neuron)
    print(connections_per_neuron)
    print(connections_per_neuron[1])

        
