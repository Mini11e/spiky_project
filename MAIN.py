import numpy as np
import model


if __name__ == "__main__":
    # Create a spiking neural network with 5 neurons
    snn = model.SNN(num_neurons=3)

    # Connect neurons with random weights
    snn.connect(0, 1, 1)
    snn.connect(1, 2, 0)

    # set input currents
    snn.set_inputs()
    
    # let the neuron run for x timesteps
    voltage, spikes = snn.simulate()

    print(voltage)
    print(spikes)


            