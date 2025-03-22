import numpy as np
import model


if __name__ == "__main__":
    # Create a spiking neural network with 5 neurons
    snn = model.SNN(num_neurons=3)

    # Connect neurons with random weights
    snn.connect(0, 1, 1)
    snn.connect(1, 2, 0)
    
    # Simulate for 10 time steps with random input currents
    for time_step in range(1):
        input_currents = [1, 1, 1]  # Random input currents for each neuron
        snn.update_network(input_currents)
        # Print neuron voltages and spike states
        for i, neuron in enumerate(snn.neurons):
            print(f"Neuron {i}: Voltage = {neuron.prev_potential:.2f}, Spiked = {neuron.spike}")
            