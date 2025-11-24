import numpy as np
import time

class Neuron:
    def __init__(self, tau=20.0, threshold=0.1, reset_voltage=0.0, resting_potential=0.0):
        self.tau = tau  # Membrane time constant
        self.threshold = threshold  # Spike threshold
        self.reset_voltage = reset_voltage  # Voltage after a spike
        self.resting_potential = resting_potential  # Resting potential
        self.prev_potential = resting_potential

        self.membrane_potential = self.resting_potential  # Current membrane potential
        self.spike = False  # Spike state

    def update(self, input_current, delta_time):
        """ Update the neuron's potential based on input current and time step. """
        # Leaky integrate-and-fire model
        dV = (self.resting_potential - self.membrane_potential + input_current) / self.tau * delta_time
        self.membrane_potential += dV

        self.prev_potential = self.membrane_potential

        if self.membrane_potential >= self.threshold:
            self.spike = True
            self.membrane_potential = self.reset_voltage  # Reset voltage after spiking
        else:
            self.spike = False
    
        

class SNN:
    def __init__(self, num_neurons):
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.connections = np.zeros((num_neurons, num_neurons))

    def connect(self, from_neuron_index, to_neuron_index, weight):
        """ Create a synaptic connection from one neuron to another. """
        self.connections[from_neuron_index, to_neuron_index] = weight

    def simulate(self, neuron_index, input_current):
        """ Provide external input to a neuron. """
        self.neurons[neuron_index].update(input_current, delta_time=1.0)

    def propagate_spikes(self):
        """ Propagate spikes through the network. """
        for i, neuron in enumerate(self.neurons):
            if neuron.spike:
                for j in range(len(self.neurons)):
                    if self.connections[i, j] > 0:  # If there is a connection
                        self.neurons[j].update(self.connections[i, j], delta_time=1.0)

    def update(self, input_currents):
        """ Update the entire network for a time step. """
        for i, current in enumerate(input_currents):
            self.simulate(i, current)
        self.propagate_spikes()


