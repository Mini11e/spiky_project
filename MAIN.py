import numpy as np
import model

######TO DO######
# cut off at certain spike count
# add refractory period?
# add different amounts of input
# simulate different patterns
# self.time_steps in simulate()

# add specifications of variable type in defs


if __name__ == "__main__":
    # Create a spiking neural network with x neurons
    timesteps = 2000

    snn = model.SNN(num_neurons=20, time_steps=timesteps)

    # Connect neurons with
    snn.auto_connect(0.15, 13, 3)

    # loc 1-1.5, scale 0-0.5, step 0.1

    # let the neuron run for x timesteps
    voltage, spikes = snn.simulate(time_steps=timesteps)

    # plot network, voltage and spikes
    snn.graph()
    snn.plot(spikes)

            