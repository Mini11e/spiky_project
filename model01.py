import numpy as np

######TO DO######
# network propagates spikes but no time distinction = everything happens at the same time -> fix
# add visualisation
# add multiple connections
# check lIF formula


class Neuron:
    def __init__(self, tau=20.0, threshold=0.1, reset_voltage=0.0, resting_potential=0.0):
        self.tau = tau  # Membrane time constant
        self.threshold = threshold  # Spike threshold
        self.reset_voltage = reset_voltage  # Voltage after a spike
        self.resting_potential = resting_potential  # Resting potential
        self.prev_potential = resting_potential

        self.membrane_potential = self.resting_potential  # Current membrane potential
        self.spike = False  # Spike state

    def update_neuron(self, input_current, delta_time):
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
        self.input_weights = np.zeros(len(self.connections))

    def connect(self, from_neuron_index, to_neuron_index, weight):
        """ Simluates a synaptic connection from one neuron to another by storing weights in an array representing the connections """
        self.connections[from_neuron_index, to_neuron_index] = weight
        self.input_weights = self.connections.sum(axis=0)
             

    def update_network(self, input_currents):
        """ Update the entire network for a time step. """
        updated_input_currents = input_currents + self.input_weights

        for i, current in enumerate(updated_input_currents):
            self.neurons[i].update_neuron(current, delta_time=1.0)


