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


if __name__ == "__main__":
    
    # set time steps (ms) for each experimental loop
    timesteps = 200

    loc = 0.85
    scale = 0.2
  
    # set up the model and connection strength
    snn = model.SNN(num_neurons=20, time_steps=timesteps, loc = loc, scale = scale, plot_xlim = [0,50])
    snn.auto_connect(0.15, 20, 3)

    for i in range(10):
        j = i*2
        k = i*2
        while j >0:
            snn.set_inputs(i, 30, 40)
            j -= 2
        voltage, spikes = snn.simulate(time_steps=timesteps)
        snn.plot(spikes, k)
    # let the neuron run for x timesteps
    
            

    # save spike train plot of each experimental loop in one image
    #plt.savefig("spiky_project/EXP_different_locs_scales_1noise/summary_plots")
    plt.close()

    
            