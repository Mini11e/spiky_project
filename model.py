import numpy as np

######TO DO######
# network propagates spikes but no time distinction = everything happens at the same time -> fix
# add visualisation
# add multiple connections
# check lIF formula
#(self, delta_time = 1.0, tau=20.0, threshold=0.1, reset_voltage=0.0, resting_potential=0.0, neurons=10, input_matrix=None, connectivity_matrix=None)


import numpy as np

class SNN:
    def __init__(self, delta_time=1.0, threshold=-55, V_noise=0.0, resting_potential=-65, tau=10.0, g_L=10.0, t_refr=2.0,
                 neurons=10, n_inputs=10, input_matrix=None, connectivity_matrix=None, w_inh=None,
                 **kwargs):
        '''
        Simulate the activity of a LIF neuron.

        Args:
            threshold     -- the threshold for neuron firing
            V_noise   -- random noise added to the signal
            delta_time        -- time step for the simulation
            neurons -- number of neurons in the network
            n_inputs  -- number of input sources to the neurons
            input_matrix     -- feedforward connectivity matrix from the input layer to LIF layer
            connectivity_matrix    -- lateral connectivity matrix within the LIF layer
        Constant params:
            g_L   -- leak of the conductance (inverse membrane resistance)
            tau -- membrane time constant: how quickly membrane voltage returns to the resting state
            resting_potential   -- resting potential (equal to voltage reset after spike)
        '''
        self.delta_time    = delta_time
        self.resting_potential   = resting_potential
        self.tau = tau
        self.g_L   = g_L

        self.threshold   = threshold
        self.V_noise = V_noise
        self.t_refr  = t_refr

        self.n_inputs  = n_inputs
        self.neurons = neurons


            
    def _delta_time(self, V, I):
        
        # Voltage update
        noise   = np.random.rand() * self.V_noise
        # insert your formula for the membrane potential (voltage) update
        #dV = noise + (-(V - self.resting_potential) + I / self.g_L) / self.tau
        dV = noise + (-(V - self.resting_potential) / self.tau + I)
        # integrate the above update
        V += dV * self.delta_time

        # refractory
        #V[refr > 0] = self.resting_potential
        #refr[refr > 0] -= 1

        fired = V > self.threshold

        V[fired]    = self.resting_potential
        #refr[fired] = self.t_refr / self.delta_time
        return V, fired
    
    def simulate(self, length, external_input=None, input_scale=None):
        '''
        Args:
            external_input -- input to the neuron
            length         -- simulation length [ms]
        '''

        delta_times    = np.arange(0, length + self.delta_time, self.delta_time)      # simulation time steps [ms]
        voltage       = np.zeros((self.neurons, len(delta_times)))  # array for saving voltage history
        voltage[:, 0] = self.resting_potential                                     # set initial voltage to resting potential
        spikes        = np.zeros((self.neurons, len(delta_times)))  # initialize spike output
        refr          = np.zeros((self.neurons,))

        # simulation
        for t in range(1, len(delta_times)):
            # calculate input to the model: a sum of the spiking inputs weighted by corresponding connections
            external_input = np.dot(self.input_matrix, external_input[:, t])
            lateral_input = np.dot(self.connectivity_matrix, spikes[:, t-1]) 
            total_input = external_input + lateral_input

            #if total_input.sum() > 0:
            #    print(t, total_input.nonzero()[0], external_input[:, t].nonzero()[0])

            # update voltage and record spikes
            voltage[:, t], spikes[:, t] = self._delta_time(voltage[:, t-1], total_input)
            
        return voltage, spikes


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
    
        

class SNN_:
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


