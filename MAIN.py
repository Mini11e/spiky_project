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
# make another heatmap showing average pattern size
# new measure for patterns ?
# try diffent noise params to underline hypothesis
# save experiments in folder: with refr/no refr, with noise clipping/no noise clipping
# ask Gordon: different noise distribution? otherwise is truncated Gaussian


if __name__ == "__main__":
    
    # set time steps (ms) for each experimental loop
    timesteps = 800

    # set arrays for different locs and scales that should be trialled
    locs = np.round(np.linspace(0.4, 1, 7), 2) #np.round(np.linspace(0.8, 1.5, 5), 2)
    scales = np.round(np.linspace(0.1, 0.8, 8), 2)
    
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
    df_isi_means = pd.DataFrame({
      'x': empty_arr,
      'y': empty_arr,
      'z': empty_arr
    })

    df_pattern_len = pd.DataFrame({
      'x': empty_arr,
      'y': empty_arr,
      'z': empty_arr
    })
     
    # subplots for summary image of experimental loop
    fig, ax = plt.subplots(nrows = len(locs), ncols = len(scales), figsize = (500, 300))
    plt.tight_layout()
    #plt.set_xlabel('Timesteps')
    #plt.set_ylabel('Neurons')
    
    for loc in locs:
        for scale in scales:
  
            # set up the model and connection strength
            snn = model.SNN(num_neurons=50, time_steps=timesteps, loc = loc, scale = scale, plot_xlim = [300,800])
            snn.auto_connect(0.05, 30, 10)

            # let the neuron run for x timesteps
            voltage, spikes = snn.simulate(time_steps=timesteps)
           
            # add rsync and spike count of this experimental loop to data frames
            rsync = snn.rsync_measure(spikes)
            spikecount = np.mean(np.sum(spikes, axis=1))
            isi_mean = snn.isi_measure(spikes)
            pattern_length = np.mean(snn.analyse_plot_patterns(spikes))

            df_rsyncs["x"][loop] = loc
            df_rsyncs["y"][loop] = scale
            df_rsyncs["z"][loop] = rsync

            df_spikes["x"][loop] = loc
            df_spikes["y"][loop] = scale
            df_spikes["z"][loop] = spikecount

            df_isi_means["x"][loop] = loc
            df_isi_means["y"][loop] = scale
            df_isi_means["z"][loop] = isi_mean

            df_pattern_len["x"][loop] = loc
            df_pattern_len["y"][loop] = scale
            df_pattern_len["z"][loop] = pattern_length

            # save spike train plot on respective axis for the summary image
            file = snn.plot_voltage_spikes(spikes)
            img = Image.open(file)
            ax[num_locs, num_scales].imshow(img)
            img.close()

            # update helper variables
            loop += 1
            num_scales += 1

        # update helper variables 
        num_scales = 0
        num_locs += 1
        
            

    # save spike train plot of each experimental loop in one image
    #plt.savefig("spiky_project/EXP_different_locs_scales_1noise/summary_plots")
    plt.close()


    snn.graph()

    # plot heatmaps of rsyncs and spike counts
    snn.synchrony_spikes_heatmaps(df_rsyncs, df_spikes, df_isi_means, df_pattern_len)

    
            