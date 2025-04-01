import numpy as np

######TO DO######
# read up on LIF formula
# figure out reasonable (LIF) formula + threshold/resting potential values
# add visualisation
# add specifications of variable type in defs


#(self, delta_time = 1.0, tau=20.0, threshold=0.1, reset_voltage=0.0, resting_potential=0.0, neurons=10, input_matrix=None, connectivity_matrix=None)
# in LIF: threshold: -55, reset_voltage: -75, then formula with -



class SNN:
    def __init__(self, delta_time=1.0, resting_potential=0.0, threshold=0.1, tau=10.0,
                 num_neurons=10, time_steps = 10, num_inputs = 10, input_matrix=None, connectivity_matrix=None):
        '''
        1) SNN that conductions spikes in an interconnected network of LIF neurons

        parameters:
            delta_time: time step for the simulation
            threshold: voltage threshold for when to spike
            resting_potential: membrane potential at rest/same as reset voltage
            tau: membrane time constant 
            num_neurons: number of connected neurons in the network
            input_matrix: matrix of external inputs for neurons (rows) per timestep (columns)
            connectivity_matrix: weight matrix of inter-connecting neurons, 0 for no connection
        '''

        self.delta_time = delta_time
        self.resting_potential = resting_potential
        self.threshold = threshold
        self.tau = tau
        self.neurons = num_neurons
        
        
        self.input_matrix = np.zeros((num_neurons, time_steps+1)) #think about if this should be an array, generate input randomly, shu
        self.connectivity_matrix = np.zeros((num_neurons, num_neurons))

        #self.t_refr = t_refr



    def connect(self, from_neuron, to_neuron, weight):
        '''
        1) Basis connectivity matrix is all zeros. This function sets individual matrix elements to chosen values.
        The matrix elements represent weights that connects neurons laterally, zeros representing n connection,
        non-zero values represent a connection. The matrix goes from each neuron(row) to each neuron(column).
        
        parameters:
            from_neuron: input neuron
            to_neuron: receiving neuron
            weight: weight of the connection
        '''

        self.connectivity_matrix[from_neuron, to_neuron] = weight



    def set_inputs(self, neuron, timestep, input_current):
        '''
        1) Basis input matrix is all zeros. This function sets individual matrix elements to chosen values. 
        The matrix elements represent input currents that each neuron(row) gets per timestep(column).

        parameters:
            neuron: neuron that gets input
            timestep: timestep t
            input_current: value of the input
        '''

        self.input_matrix[neuron, timestep] = input_current



    def lif_integration(self, membrane_potentials, input_currents):
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
    

    
    def simulate(self, time_steps, external_input_matrix=None):
        '''
        1) calculates spike input per timestep as a sum of external inputs + lateral inputs of previous timestep
        uses dot product of connectivity matrices
        2) records voltages and spikes for each timestep in arrays

        parameters:
            external_input_matrix: matrix of external input to the neuron
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
            #external_input = np.dot(self.input_matrix, external_input[:, t])
            external_input = self.input_matrix[:, t-1] # external input for each neuron at given timestep, take from matrix: all rows of column=t
            lateral_input = np.dot(self.connectivity_matrix, spikes[:, t-1]) # matrix*array=array
            total_input = external_input + lateral_input #arr+arr=arr

            # record voltage and spikes
            voltage[:, t], spikes[:, t] = self.lif_integration(voltage[:, t-1], total_input)

            # add visualise() per timestep
            
        return voltage, spikes
    
    def visualise():
        # heatmap for voltages
        # spikes as horizontal eventplot?
        pass
    
