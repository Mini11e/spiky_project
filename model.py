import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random


#(self, delta_time = 1.0, tau=20.0, threshold=0.1, reset_voltage=0.0, resting_potential=0.0, neurons=10, input_matrix=None, connectivity_matrix=None)
# in LIF: threshold: -55, reset_voltage: -75, then formula with -



class SNN:
    def __init__(self, delta_time=1.0, resting_potential=-65, threshold=-55, tau=10.0,
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
        self.all_voltages = np.zeros((num_neurons, 2*(time_steps)))
        self.time_steps = time_steps
        
        
        self.input_matrix = np.zeros((num_neurons, time_steps+1)) #think about if this should be an array, generate input randomly, shu
        self.connectivity_matrix = np.zeros((num_neurons, num_neurons))

        # good noise parameters: loc=0.85, scale = 0.2
        self.noise = np.random.normal(loc=0.85, scale=0.2, size=(self.neurons, time_steps)).clip(0, None)

        #self.t_refr = t_refr

        random.seed(30)
        self.neuron_colours = []
        for i in range(self.neurons):
            self.neuron_colours.append('#%06X' % random.randint(0, 0xFFFFFF))




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

        self.connectivity_matrix[to_neuron, from_neuron] = weight



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



    def lif_integration(self, V, input_currents, t):
        '''
        1) uses the LIF formula to calculate how spikes are integrated

        parameters:
            V: array of membrane potentials of all neurons
            input_currents: array of input currents to each neuron as a sum of lateral and external input

        returns:
            V: array of voltages of all neurons after spikes
            spiked: array of booleans of whether or not the neuron spiked

        LIF formula: tau * dV/dt = -(V - E_L) + I/g_L
        LIF variables: tau= membrane time constant, dV= voltage to be conducted, dt= delta time= time step for the simulation
                        V= membrane potential, E_L= resting_potential, I= input current, g_L= leak conductance
        '''
        # Voltage update
        # dV = -(membrane_potential - self.resting_potential) + input_current / self.g_L / self.tau #LIF
        # dV = noise + (-(V - self.resting_potential) / self.tau + I) # viktoriias
        # V += dV * self.dt #viktoriias
        dV = self.noise[:,t-1] + (-(V - self.resting_potential) + input_currents) / self.tau * self.delta_time
        V += dV * self.delta_time

        self.all_voltages[:,(2*t)-2] = V
    
        #refractory
        #V[refr > 0] = self.resting_potential
        #refr[refr > 0] -= 1

        # if the neuron spiked, reset its voltage to the resting potential
        spiked = V > self.threshold
        V[spiked]    = self.resting_potential
        self.all_voltages[:,(2*t)-1] = V

        #refr[spiked] = self.t_refr / self.delta_time

        return V, spiked
    

    
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
            voltage[:, t], spikes[:, t] = self.lif_integration(voltage[:, t-1], total_input, t)

            # add visualise() per timestep
            
        return voltage, spikes
    
        
    
    def plot(self, spikes):
        # heatmap for voltages
        # spikes as horizontal eventplot?

        print(self.all_voltages)
            
        fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = True)
        
        fig.set_size_inches(10, 5)
        dim1 = np.linspace(0, self.time_steps-1, self.time_steps*2)

        ax[0].hlines(y = self.threshold, xmin = 0, xmax = self.time_steps, colors = "red", linestyles = "dashed", label = "Threshold")
        
        # subplot for each neuron
        for i in range(self.neurons):

            colors = ["green", "blue", "orange", "yellow", "lightblue", "lightgreen", "purple", "lightyellow", "darkgreen", "violet"]
            

            ax[0].plot(dim1, self.all_voltages[i], label = f"Neuron {i}", color = self.neuron_colours[i])
            ax[0].set_xlabel('Timesteps')
            ax[0].set_ylabel('Voltage')
            ax[0].set_title(f"Voltages")
            #ax[0].legend(loc = "upper left")

            j = 0
            plot_spikes = np.zeros(len(spikes[i]))
            for k in range(len(spikes[i])):

                if spikes[i,k] == 1:
                    plot_spikes[j] = k
                    j += 1
            plot_spikes = plot_spikes[plot_spikes != 0]
                
            print("plotspikes")
            print(plot_spikes)
            ax[1].eventplot(plot_spikes, label = f"Neuron {i}", lineoffsets = i, linelengths= 0.5, color = self.neuron_colours[i])
            ax[1].set_xlabel('Timesteps')
            ax[1].set_ylabel('Spikes')
            ax[1].set_title(f"Spikes")
            ax[1].legend()
        
            
        
        #fig.suptitle(f'Metrics: tau={self.tau}, thresh={self.threshold}')
        plt.show()

    def graph(self):

        seed = 30133  # Seed random number generators for reproducibility

        g = nx.DiGraph()
        
        connected_nodes = []   
        
        for i in range(self.neurons):
            for j in range(self.neurons):
                if self.connectivity_matrix[i][j] != 0:
                    g.add_edge(j, i, weight = self.connectivity_matrix[i][j])
                    connected_nodes.append(i)
                    connected_nodes.append(j)
        
        for k in range(self.neurons):
           if k not in connected_nodes:
              g.add_node(k)

        colours = {}
        for m in range(self.neurons):
            colours[m] = self.neuron_colours[m]

        # pos = nx.spring_layout(g, k = 4, seed=seed)
        pos = nx.circular_layout(g)
            
        nx.draw(g, node_color=[colours[node] for node in g.nodes], pos=pos, with_labels=True)
                
        plt.show()


    
