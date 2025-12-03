import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import seaborn as sns
from scipy.interpolate import make_smoothing_spline, BSpline


#(self, delta_time = 1.0, tau=20.0, threshold=0.1, reset_voltage=0.0, resting_potential=0.0, neurons=10, input_matrix=None, connectivity_matrix=None)
# in LIF: threshold: -55, reset_voltage: -75, then formula with -
# change to give parameters in main: loc, scale, plot range



class SNN:
    def __init__(self, delta_time=1.0, resting_potential=-65, threshold=-55, tau=10.0, t_refr = 0,
                 num_neurons=10, time_steps = 10, num_inputs = 10, input_matrix=None, connectivity_matrix=None, max_spikes_record=100000, loc = 0.85, scale = 0.2, plot_xlim = [0, 2000]):
        '''
        SNN that conductions spikes in an interconnected network of LIF neurons

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
        self.t_refr = t_refr

        self.max_spikes_record = max_spikes_record
        self.spikes_num = 0        
        
        self.input_matrix = np.zeros((num_neurons, time_steps+1))
        self.connectivity_matrix = np.zeros((num_neurons, num_neurons))

        self.loc = loc
        self.scale = scale
        self.plot_xlim = plot_xlim

        # good noise parameters: loc=0.85, scale = 0.2 # .clip(0, None) to remove anything below 0
        self.noise = np.random.normal(loc=self.loc, scale=self.scale, size=(self.neurons, time_steps))#.clip(0, None)
        #self.noise = np.random.gamma(shape=self.loc, scale=self.scale, size=(self.neurons, time_steps))
        

        # self.t_refr = t_refr
        
        # colour scheme seed
        random.seed(30)
        self.neuron_colours = []
        for i in range(self.neurons):
            self.neuron_colours.append('#%06X' % random.randint(0, 0xFFFFFF))




    def connect(self, from_neuron, to_neuron, weight):
        '''
        Basis connectivity matrix is all zeros. This function sets individual matrix elements to chosen values.
        The matrix elements represent weights that connects neurons laterally, zeros representing n connection,
        non-zero values represent a connection. The matrix goes from each neuron(column) to each neuron(row).
        
        parameters:
            from_neuron: input neuron
            to_neuron: receiving neuron
            weight: weight of the connection
        '''

        self.connectivity_matrix[to_neuron, from_neuron] = weight

    
    def auto_connect(self, percentage, weight, max_inputs = 1000):
        '''
        This function fills the connectivity matrix randomly and automatically, according to input percentage.
        The function also can also restrict amount of input connections per neuron
        
        parameters:
            percentage: connectivity percentage
            weight: weight of the connections
            max_inputs: maximum input connections per neuron
        '''
        #seed: 30
        #seed: 105
        
        count_inputs = np.zeros(self.neurons)
        distribute_func = lambda m, n: (lambda base, remainder: [base + (1 if i < remainder else 0)for i in range(n)])(m // n,m % n)
        connections_per_neuron = distribute_func((round(self.neurons*percentage*self.neurons)), self.neurons)
        np.random.seed(30)
        np.random.shuffle(connections_per_neuron)

        np.random.seed(30)
        for i in range(self.neurons):
            for j in range(connections_per_neuron[i]):
                input_assigned = False
                while input_assigned == False:
                    random_neuron = random.randint(0, self.neurons-1)

                    if random_neuron !=i and count_inputs[random_neuron] <= max_inputs-1:
                        self.connect(i, random_neuron, weight)
                        count_inputs[random_neuron] += 1
                        input_assigned = True                   



    def set_inputs(self, neuron, timestep, input_current):
        '''
        Basis input matrix is all zeros. This function sets individual matrix elements to chosen values. 
        The matrix elements represent input currents that each neuron(row) gets per timestep(column).

        parameters:
            neuron: neuron that gets input
            timestep: timestep t
            input_current: value of the input
        '''

        self.input_matrix[neuron, timestep] = input_current



    def lif_integration(self, V, input_currents, t, refr):
        '''
        Uses the LIF formula to calculate how spikes are integrated.

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

        # if the neuron spiked, reset its voltage to the resting potential
        
        #self.spikes_num += sum(spiked)
        #print(self.spikes_num)

        V[refr>0] = self.resting_potential
        refr[refr>0] -= 1

        spiked = V > self.threshold
        V[spiked]    = self.resting_potential
        self.all_voltages[:,(2*t)-1] = V

        refr[spiked] = self.t_refr / self.delta_time

        #refr = 0 #comment out if refr needed

        return V, spiked, refr
    

    
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
        start_noise = np.random.normal(loc=-65, scale=2, size=(self.neurons))
        voltage[:, 0] = start_noise #self.resting_potential               # set initial voltage to resting potential
        spikes        = np.zeros((self.neurons, len(steps)))  # initialise spike output
        refr          = np.zeros((self.neurons,))
    
        # simulation
        for t in range(1, len(steps)):
            if self.spikes_num < self.max_spikes_record:
                # calculate input to the model: a sum of the spiking inputs weighted by corresponding connections
                #external_input = np.dot(self.input_matrix, external_input[:, t])
                external_input = self.input_matrix[:, t-1] # external input for each neuron at given timestep, take from matrix: all rows of column=t
                lateral_input = np.dot(self.connectivity_matrix, spikes[:, t-1]) # matrix*array=array
                total_input = external_input + lateral_input #arr+arr=arr

                # record voltage and spikes
                voltage[:, t], spikes[:, t], refr = self.lif_integration(voltage[:, t-1], total_input, t, refr)
            
        return voltage, spikes
    

    def rsync_measure(self, firings):
        '''

        Implements the RSYNC formula.

        parameters:
            firings: matrix of spike trains per experimental loop

        '''

        def exp_convolve(spike_train):
            tau = 3.0  # ms
            exp_kernel_time_steps = np.arange(0, tau * 10, 1)
            decay = np.exp(-exp_kernel_time_steps / tau)
            exp_kernel = decay
            return np.convolve(spike_train, exp_kernel, 'same')

        firings = np.apply_along_axis(exp_convolve, 1, firings)
        num = np.var(np.mean(firings, axis=0))  # spatial mean across cells, at each time
        den = np.mean(np.var(firings, axis=1))  # variance over time of each cell
        return num / (den + 1e-100)
    

    def isi_measure(self, spikes):

        isis_all = []
        for s in spikes:
                
            isis_neuron = []
            counter = 0

            for t in s:
                    
                if t == 0:
                    counter += 1

                if t == 1:
                    isis_neuron.append(counter)
                    counter = 0
                    

            isis_all.append(np.mean(isis_neuron))
        
        isis_all_cleaned = [x for x in isis_all if str(x) != 'nan']

        return np.mean(isis_all_cleaned)

    
        
    
    def plot_voltage_spikes(self, spikes):
        '''
        Plots the spikes (and voltages) of each neuron and saves it as an image in a folder.
        
        parameters:
            spikes: matrix of spike trains
        '''
        fig, ax = plt.subplots(nrows = 1, ncols = 2, sharex = True)
        #fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True)

        fig.set_size_inches(10, 5)
        dim1 = np.linspace(0, self.time_steps-1, self.time_steps*2)

        ax[0].hlines(y = self.threshold, xmin = 0, xmax = self.time_steps, colors = "red", linestyles = "dashed", label = "Threshold")
        
        # subplot for each neuron
        for i in range(self.neurons):            

            
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
            
            ax[1].eventplot(plot_spikes, label = f"Neuron {i}", lineoffsets = i, linelengths= 0.5, color = self.neuron_colours[i])
            #ax[1].scatter(spikes[1], spikes[0])
            #ax.set_xlabel('Timesteps')
            #ax.set_ylabel('Neurons')
            #ax.set_title(f"Spikes")
            ax[1].legend(loc = "upper left", prop={'size': 6})
            ax[1].set_xlim(self.plot_xlim)

            # if scatterplot:
            # neuron_idx, time_idx = np.where(spike_trains == 1)
            # plt.scatter(time_idx, neuron_idx, s=5, marker='*')
        
        # Save plot as image
        fig.suptitle(f'Gaussian Noise Parameters: Loc={self.loc:.2f}, Scale={self.scale:.2f}')
        title = f'spiky_project/EXP_different_locs_scales_1noise/SPIKES_loc{self.loc:.2f}_scale{self.scale:.2f}.png'
        plt.savefig(title)
        #plt.show()
        plt.close()

        return title

    def analyse_plot_patterns(self, spikes):

        patterns = spikes.sum(axis=0)
        plt.plot(range(self.time_steps+1), patterns)


        # Save plot as image
        plt.title(f'Gaussian Noise Parameters: Loc={self.loc:.2f}, Scale={self.scale:.2f}')
        title = f'spiky_project/EXP_different_locs_scales_1noise/PATTERNS_loc{self.loc:.2f}_scale{self.scale:.2f}.png'
        plt.savefig(title)
        #plt.show()
        plt.close()

        max = np.max(patterns)
        pattern_threshold = max*0.2 # = 0
        len_patterns = []
        counter = 0

        for p in patterns:

            if p <= pattern_threshold:
                len_patterns.append(counter)
                counter = 0

            if p > pattern_threshold:
                counter +=1
        
        len_patterns = np.asarray(len_patterns)
        len_patterns = len_patterns[len_patterns != 0]
        return(len_patterns)



    def graph(self):
        '''
        Draws a circular graph that shows how neurons are connected.

        '''
        # Initialise graph and list for nodes
        g = nx.DiGraph()
        connected_nodes = []   
        
        # Connect nodes according to connectivity matrix
        for i in range(self.neurons):
            for j in range(self.neurons):
                if self.connectivity_matrix[i][j] != 0:
                    g.add_edge(j, i, weight = self.connectivity_matrix[i][j])
                    connected_nodes.append(i)
                    connected_nodes.append(j)
        
        for k in range(self.neurons):
           if k not in connected_nodes:
              g.add_node(k)

        # Set colours to match the neurons in other plots
        colours = {}
        for m in range(self.neurons):
            colours[m] = self.neuron_colours[m]

        # Circular layout
        pos = nx.circular_layout(g)
        
        # Draw graph and save as file
        fig = plt.figure()
        nx.draw(g, node_color=[colours[node] for node in g.nodes], pos=pos, with_labels=True)
        #plt.show()     
        plt.savefig("spiky_project/EXP_different_locs_scales_1noise/graph")
        plt.close()


    def synchrony_spikes_heatmaps(self, df1, df2, df3, df4, interconnection, noise, refr, patternmethod):
        '''
        Plots two heatmaps, one with synchrony measured with RSYNC and one with spike counts.

        '''
        sns.set_theme()
    
        # Preprocess data frames
        df1 = (df1.pivot(index="y", columns="x", values="z"))
        df2 = (df2.pivot(index="y", columns="x", values="z"))
        df3 = (df3.pivot(index="y", columns="x", values="z"))
        df4 = (df4.pivot(index="y", columns="x", values="z"))

        # Two heatmaps with the numeric values with heatmap1 for rsync values and heatmap2 for spike counts
        fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize=(12, 10))
        fig.suptitle(f'Interconnection={interconnection} Noise={noise} Refr={refr} Patternmethod={patternmethod}')
        
        heatmap1 = sns.heatmap(data = df1, annot = True, fmt=".2f", linewidths=.5, ax=ax[0][0], cmap = sns.color_palette("YlOrBr", as_cmap=True))
        heatmap1.set(xlabel="mean", ylabel="variance")
        heatmap1.set_title("Rsync")
        heatmap2 = sns.heatmap(data = df2, annot = True, fmt="1.0f", linewidths=.5, ax=ax[0][1], cmap = sns.color_palette("BuGn", as_cmap=True))
        heatmap2.set(xlabel="mean", ylabel="variance")
        heatmap2.set_title("Average Spike Count")
        heatmap3 = sns.heatmap(data = df3, annot = True, fmt=".2f", linewidths=.5, ax=ax[1][0], cmap = sns.color_palette("Blues_d", as_cmap=True))
        heatmap3.set(xlabel="mean", ylabel="variance")
        heatmap3.set_title("Average ISI")
        heatmap4 = sns.heatmap(data = df4, annot = True, fmt=".2f", linewidths=.5, ax=ax[1][1], cmap = sns.color_palette("Purples", as_cmap=True))
        heatmap4.set(xlabel="mean", ylabel="variance")
        heatmap4.set_title("Average Pattern Length")
        heatmap1.invert_yaxis()
        heatmap2.invert_yaxis()
        heatmap3.invert_yaxis()
        heatmap4.invert_yaxis()
        fig.tight_layout()

        # Save heatmap figure as image
        plt.savefig(f'spiky_project/EXP_different_locs_scales_1noise/HEATMAP_interconnection{interconnection}_noise{noise}_refr{refr}_patternmethod{patternmethod}.png')



    
    


    
