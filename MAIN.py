import numpy as np
import model

######TO DO######
# fix plotting of spikes
# plot regular vltage instead of all voltages

# create larger interconnected network
# add specifications of variable type in defs


if __name__ == "__main__":
    # Create a spiking neural network with x neurons
    snn = model.SNN(num_neurons=10, time_steps=50)

    # Connect neurons with random weights
    snn.connect(1, 2, 30)
    snn.connect(1, 4, 30)
    snn.connect(2, 3, 30)
    snn.connect(4, 3, 30)
    snn.connect(4, 5, 30)
    snn.connect(3, 6, 30)
    snn.connect(6, 0, 30)

    # set input currents
    snn.set_inputs(1,25,30)
    snn.set_inputs(1,0,0)
    snn.set_inputs(2,0,0)

    
    # let the neuron run for x timesteps
    voltage, spikes = snn.simulate(time_steps=50)

    print(voltage)
    print(spikes)
    print(spikes[0])

    snn.plot(spikes)

            