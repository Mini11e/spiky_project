import numpy as np
import random
import MAIN
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline, BSpline


if __name__ == "__main__":

    time_steps = range(10)

    x1 = np.array(((0,0,0,1,0,1,0,0,0,0),(0,0,0,1,0,1,0,0,0,0),(0,0,0,1,0,1,0,0,0,0)))
    patterns = x1.sum(axis=0)
    print(np.sort(patterns))

    x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/16)
    rng = np.random.default_rng()
    y =  np.sin(x) + 0.4*rng.standard_normal(size=len(x))

    print(x)
    print(y)

    xnew = np.arange(0, 9/4, 1/50) * np.pi
    spl = make_smoothing_spline(time_steps, patterns, lam=0.02)
    plt.plot(xnew, spl(xnew), '-.', label=fr'$\lambda=${0.02}')
        
    plt.plot(time_steps, patterns, 'o')

    plt.show()
    
    #plt.plot(time_steps, patterns)
    #plt.show()
        
