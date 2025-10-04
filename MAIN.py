import numpy as np
import model
import pandas as pd

######TO DO######
# add refractory period?
# add different amounts of input?
# simulate different patterns

# add specifications of variable type in defs


if __name__ == "__main__":
    # Create a spiking neural network with x neurons
    timesteps = 2000

    locs = [1.0, 1.5] #[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    scales = [0.1,0.5] # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rsyncs = np.zeros(len(locs)*len(scales))
    counter = 0

    df = pd.DataFrame({
      'x': rsyncs,
      'y': rsyncs,
      'z': rsyncs
    })

    for s in locs:
        for t in scales:

            snn = model.SNN(num_neurons=20, time_steps=timesteps, loc = s, scale = t, plot_xlim = [1800,2000])
            snn.auto_connect(0.15, 13, 3)
            # let the neuron run for x timesteps
            voltage, spikes = snn.simulate(time_steps=timesteps)
            rsync = snn.rsync_measure(spikes)
            print("STEAK")
            print(rsync)
            df["x"][counter] = s
            df["y"][counter] = t
            df["z"][counter] = rsync

            counter += 1
            
            snn.plot(spikes)
            #snn.graph()

    print("NOSTEAK")
    print(df.head())        

    snn.plot_synchrony(df)
            

    
    

    # loc 1-1.5, scale 0-0.5, step 0.1

    
    # plot network, voltage and spikes
    

            