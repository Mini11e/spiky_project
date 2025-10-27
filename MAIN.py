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
    
    # set time steps (ms) for each experimental loop
    timesteps = 2000

    # set arrays for different locs and scales that should be trialled
    locs = np.round(np.linspace(0.8, 1.5, 5), 2)
    scales = np.round(np.linspace(0.1, 0.5, 5), 2)
    
    # helper variables
    loop = 0
    num_locs = 0
    num_scales = 0

    # data frames to save rsyncs and spike counts
    empty_arr = np.zeros(len(locs)*len(scales))
    df_rsyncs = pd.DataFrame({
      'x': empty_arr,
      'y': empty_arr,
      'z': empty_arr
    })
    df_spikes = pd.DataFrame({
      'x': empty_arr,
      'y': empty_arr,
      'z': empty_arr
    })
     
    # subplots for summary image of experimental loop
    fig, ax = plt.subplots(nrows = len(locs), ncols = len(scales), figsize = (500, 300))
    plt.tight_layout()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Neurons')
    
    for loc in locs:
        for scale in scales:
  
            # set up the model and connection strength
            snn = model.SNN(num_neurons=20, time_steps=timesteps, loc = loc, scale = scale, plot_xlim = [1800,2000])
            snn.auto_connect(0.15, 13, 3)

            # let the neuron run for x timesteps
            voltage, spikes = snn.simulate(time_steps=timesteps)
           
            # add rsync and spike count of this experimental loop to data frames
            rsync = snn.rsync_measure(spikes)
            spikecount = np.sum(spikes)
          
            df_rsyncs["x"][loop] = loc
            df_rsyncs["y"][loop] = scale
            df_rsyncs["z"][loop] = rsync

            df_spikes["x"][loop] = loc
            df_spikes["y"][loop] = scale
            df_spikes["z"][loop] = spikecount

            # save spike train plot on respective axis for the summary image
            file = snn.plot(spikes)
            img = Image.open(file)
            ax[num_locs, num_scales].imshow(img)

            # update helper variables
            loop += 1
            num_scales += 1

        # update helper variables 
        num_scales = 0
        num_locs += 1
            

    # save spike train plot of each experimental loop in one image
    plt.savefig("spiky_project/experiments/summary_plots")

    #snn.graph()

    # plot heatmaps of rsyncs and spike counts
    snn.synchrony_spikes_heatmaps(df_rsyncs, df_spikes)

    
            