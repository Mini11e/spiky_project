import numpy as np

######TO DO######
# add def to build input matrix or just an array
# understand what the dot product is doing
# add a main and test SNN
# add multiple connections/def for establishing connection matrix?
# add visualisation
# add specifications of variable type in defs
# play with parameters 


#(self, delta_time = 1.0, tau=20.0, threshold=0.1, reset_voltage=0.0, resting_potential=0.0, neurons=10, input_matrix=None, connectivity_matrix=None)
# in LIF: threshold: -55, reset_voltage: -75, then formula with -



class SNN:
    def __init__(self, delta_time=1.0, resting_potential=-65, threshold=-55, tau=10.0,
                 num_neurons=10, input_matrix=None):
        '''
        1) SNN that conductions spikes in an interconnected network of LIF neurons

        parameters:
            delta_time: time step for the simulation
            threshold: voltage threshold for when to spike
            resting_potential: membrane potential at rest/same as reset voltage
            tau: membrane time constant 
            num_neurons: number of connected neurons in the network
            input_matrix: good question??? weight matrix for how to forward external input, for me probably just 1 for each neuron that should be a staring neuron, 0 for all others
            connectivity_matrix: weight matrix of inter-connecting neurons, 0 for no connection
        '''

        self.delta_time = delta_time
        self.resting_potential = resting_potential
        self.threshold = threshold
        self.tau = tau
        self.neurons = num_neurons
        
        
        self.input_matrix = input_matrix(num_inputs)
        self.connectivity_matrix = np.zeros((num_neurons, num_neurons)) 

        #self.t_refr = t_refr



    def connect(self, from_neuron, to_neuron, weight):
        '''
        1) adds lateral connections to the network i.e. substitutes zeros (representing no connection)
        of the connectivitiy matrix by non-zero weights

        parameters:
            from_neuron: input neuron
            to_neuron: receiving neuron
            weight: weight of the connection
        '''

        self.connectivity_matrix[from_neuron, to_neuron] = weight



    def adjust_input_matrix():
        pass



    def lif_integration(self, membrane_potentials, input_currents): #array or singular current?
        '''
        1) uses the LIF formula to calculate how spikes are integrated

        parameters:
            membrane_potentials: array of membrane potentials of all neurons
            input_currents: array of input currents to each neuron as a sum of lateral and external input

        returns:
            dV: array of current voltages of all neurons
            spiked: array of booleans of whether or not the neuron spiked

        LIF formula: tau * dV/dt = -(V - E_L) + I/g_L
        LIF variables: tau= membrane time constant, dV= voltage to be conducted, dt= delta time= time step for the simulation
                        V= membrane potential, E_L= resting_potential, I= input current, g_L= leak conductance
        '''
        
        # Voltage update
        # dV = -(membrane_potential - self.resting_potential) + input_current / self.g_L / self.tau #LIF
        # dV = noise + (-(V - self.resting_potential) / self.tau + I) # viktoriias
        dV = (-(membrane_potentials - self.resting_potential) + input_currents) / self.tau * self.delta_time

        #refractory
        #V[refr > 0] = self.resting_potential
        #refr[refr > 0] -= 1

        # if the neuron spiked, reset its voltage to the resting potential
        spiked = dV > self.threshold
        dV[spiked]    = self.resting_potential

        #refr[spiked] = self.t_refr / self.delta_time

        return dV, spiked
    

    
    def simulate(self, time_steps, external_input=None):
        '''
        1) calculates spike input per timestep as a sum of external inputs + lateral inputs of previous timestep
        uses dot product of connectivity matrices
        2) records voltages and spikes for each timestep in arrays

        parameters:
            external_input: external input to the neuron
            time_steps: simulation time steps [ms]

        returns:
            voltages: array of voltages per neuron
            spikes: array of spikes per neuron #i think in 0 and 1, but maybe boolean?
        '''

        steps    = np.arange(0, time_steps + self.delta_time, self.delta_time)      # simulation time steps [ms] ## why +self.delta_time? one more i guess
        voltage       = np.zeros((self.neurons, len(steps)))  # array for saving voltage history
        voltage[:, 0] = self.resting_potential                # set initial voltage to resting potential
        spikes        = np.zeros((self.neurons, len(steps)))  # initialise spike output
        #refr          = np.zeros((self.neurons,))

        # simulation
        for t in range(1, len(steps)):
            # calculate input to the model: a sum of the spiking inputs weighted by corresponding connections
            external_input = np.dot(self.input_matrix, external_input[:, t]) ## how does this work? do i need this, or easier with just making it one array? I assume external inputs is the spikes and the matrix the weighting, should be all 1 for me
            lateral_input = np.dot(self.connectivity_matrix, spikes[:, t-1]) # matrix*array=array
            total_input = external_input + lateral_input #should be an array; arr+arr=arr

            # record voltage and record spikes
            
            voltage[:, t], spikes[:, t] = self.lif_integration(voltage[:, t-1], total_input)
            
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


