import numpy as np
import model


if __name__ == "__main__":
    # Create a spiking neural network with 5 neurons
    snn = model.SNN(num_neurons=3, time_steps=10)

    # Connect neurons with random weights
    snn.connect(0, 1, 0)
    snn.connect(1, 2, 5)

    # set input currents
    snn.set_inputs(0,0,20)
    snn.set_inputs(1,0,1)
    snn.set_inputs(2,0,0)
    
    # let the neuron run for x timesteps
    voltage, spikes = snn.simulate(time_steps=3)

    print(voltage)
    print(spikes)


            