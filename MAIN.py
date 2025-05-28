import numpy as np
import model

######TO DO######
# add refractory period
# add different amounts of input
# simulate different patterns

# add specifications of variable type in defs


if __name__ == "__main__":
    # Create a spiking neural network with x neurons
    snn = model.SNN(num_neurons=20, time_steps=50)

    # Connect neurons with
    snn.auto_connect(0.15, 30)

    # set input currents
    snn.set_inputs(1,25,30)

    # let the neuron run for x timesteps
    voltage, spikes = snn.simulate(time_steps=50)

    # plot network, voltage and spikes
    snn.graph()
    snn.plot(spikes)

            