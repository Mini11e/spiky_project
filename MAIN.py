import numpy as np
import model

######TO DO######
# play around with inputs that provoke a spike
# create larger interconnected network
# add specifications of variable type in defs


if __name__ == "__main__":
    # Create a spiking neural network with x neurons
    snn = model.SNN(num_neurons=3, time_steps=3)

    # Connect neurons with random weights
    #snn.connect(0, 1, 20)
    snn.connect(0, 2, 100)
    snn.connect(1, 2, 300)

    # set input currents
    snn.set_inputs(0,0,20)
    snn.set_inputs(1,0,200)
    snn.set_inputs(2,1,200)

    
    # let the neuron run for x timesteps
    voltage, spikes = snn.simulate(time_steps=3)

    print(voltage)
    print(spikes)

    snn.plot(spikes)

            