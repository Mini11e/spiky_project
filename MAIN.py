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
    timesteps = 200

    snn = model.SNN(num_neurons=20, time_steps=timesteps)

    # Connect neurons with
    snn.auto_connect(0.125, 13)
    snn.connect(13, 6, 13)
    snn.connect(16, 2, 13)

    # set input currents
    '''
    snn.set_inputs(1, 25, 40)
    snn.set_inputs(3, 25, 40)
    snn.set_inputs(5, 25, 40)
    snn.set_inputs(7, 25, 40)
    snn.set_inputs(9, 25, 40)
    snn.set_inputs(11, 25, 40)
    snn.set_inputs(13, 25, 40)
    snn.set_inputs(15, 25, 40)
    snn.set_inputs(17, 25, 40)
    snn.set_inputs(19, 25, 40)'''

    # let the neuron run for x timesteps
    voltage, spikes = snn.simulate(time_steps=timesteps)

    # plot network, voltage and spikes
    snn.graph()
    snn.plot(spikes)

            