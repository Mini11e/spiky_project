import numpy as np
import model
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cbook as cbook
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path


######TO DO######
# add refractory period?
# add different amounts of input?
# simulate different patterns

# add specifications of variable type in defs

#heatmap spike counts per locs and scales


if __name__ == "__main__":
    # Create a spiking neural network with x neurons
    timesteps = 2000

    locs = np.linspace(0.8, 1.5, 5)#[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    scales = np.linspace(0.1, 0.5, 5)#[0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rsyncs = np.zeros(len(locs)*len(scales))
    counter = 0
    lenlocs = 0
    lenscales = 0

    df = pd.DataFrame({
      'x': rsyncs,
      'y': rsyncs,
      'z': rsyncs
    })

    df2 = pd.DataFrame({
      'x': rsyncs,
      'y': rsyncs,
      'z': rsyncs
    })

    fig, ax = plt.subplots(nrows = len(locs), ncols = len(scales))
    fig.set_figwidth(50)
    fig.set_figheight(30)
  
    
    for s in locs:
        for t in scales:

            snn = model.SNN(num_neurons=20, time_steps=timesteps, loc = s, scale = t, plot_xlim = [1800,2000])
            snn.auto_connect(0.15, 13, 3)
            # let the neuron run for x timesteps
            voltage, spikes = snn.simulate(time_steps=timesteps)
           
            rsync = snn.rsync_measure(spikes)
          
            df["x"][counter] = s
            df["y"][counter] = t
            df["z"][counter] = rsync

            df2["x"][counter] = s
            df2["y"][counter] = t
            df2["z"][counter] = rsync

            title = snn.plot(spikes)

            img = Image.open(title)
            ax[lenlocs, lenscales].imshow(img)
          
            counter += 1
            
            lenscales += 1
        lenscales = 0
        lenlocs += 1
            
      
    plt.savefig("spiky_project/experiments/tessst")
    snn.plot_synchrony(df)
    snn.graph()
            

    
    

    # loc 1-1.5, scale 0-0.5, step 0.1

    
    # plot network, voltage and spikes
    

            