
# global imports

import sys, os
import matplotlib.pyplot as plt
import fabio
import h5py


# local imports

def save_plot(fig, savename):

    print "saving plot as: \n%s" %savename    

    if savename.endswith('.edf'):
        
    
        plt.savefig(savename, transparent = True, bbox_incens = "tight")
