from __future__ import division

from builtins import range
from past.utils import old_div
import sys, os
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from simplecalc.calc import define_a_line_as_mask, normalize_self
from simplecalc.slicing import troi_to_slice
import math
### side project ###
### making a alpha channel colormap:
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main():
    # Choose colormap
    cmap = pl.cm.gray_r

    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:,-1] = np.linspace(1,0, cmap.N)

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)



    ### data
    fname2015 = '/tmp_14_days/johannes1/results/mg01_5_4_3/results/mg01_5_4_3/mg01_5_4_3.replace.h5'
    h5f2015 = h5py.File(fname2015,'r')

    xbic2015 = np.asarray(h5f2015['counters/zap_p201_IC/data'])[::-1,::-1,:]
    ga2015 = np.asarray(h5f2015['detectorsum/Ga-K/data'])[::-1,::-1,:]
    as2015 = np.asarray(h5f2015['detectorsum/As-K/data'])[::-1,::-1,:]
    ni2015 = np.asarray(h5f2015['detectorsum/Ni-K/data'])[::-1,::-1,:]

    ### positions

    h5f2015.close()



    # for i, scanno in enumerate([0,2,3,4,5,6,9,10,11,12]):
    #     fig, ax1 = plt.subplots()

    #     print ('scan {}, with bias {}'.format(scanno, bias2015[i]))
    #     ax1.matshow(np.where(mask2015,ga2015[:,:,scanno],0))
    #     ax1.matshow(mask2015,cmap=my_cmap)

    #     plt.show()




    safety  = np.where(ga2015+as2015==0,1,0)
    garatio = (old_div(ga2015,(ga2015+as2015+safety)))
    gaas = np.zeros(shape=(ga2015.shape))
    xbic_norm = np.zeros(shape=(ga2015.shape))
    for i in range(len(garatio[1,1,:])):
        gaas[:,:,i]      = normalize_self((ga2015[:,:,i] + as2015[:,:,i]))
        xbic_norm[:,:,i] = normalize_self(xbic2015[:,:,i])


    ## colormap:
    cmap = pl.cm.gnuplot2
    my_cmap = cmap(np.arange(25,cmap.N))


    # Create new colormap
    my_cmap = ListedColormap(my_cmap)   
    fig, (ax1)= plt.subplots()

    troi1  = ((20,20),(50,38)) 
    slice1 = troi_to_slice(troi1)
    together = np.vstack([garatio[:,:,5][slice1][::-1,::-1],gaas[:,:,5][slice1][::-1,::-1],xbic_norm[:,:,5][slice1][::-1,::-1]])
    
    im1 = ax1.matshow(together[:,:],vmin = 0, vmax =1, cmap = my_cmap)
    x1ticklabels = []
    
    
    ax1.set_xlim([0,35])
    yticklabels = []
    for tick in ax1.get_yticks():
        yticklabels.append("{}".format((tick%50) * 20))    
    ax1.set_yticklabels(yticklabels[::-1])
    
 
    xticklabels = []
    for tick in ax1.get_xticks():
        xticklabels.append("{}".format((tick) * 20))    
#    ax1.set_xticklabels(xticklabels)

    ax1.set_xticks([])
    ax1.set_yticks([])

    ax1.vlines((troi1[1][1],troi1[1][1]*2),ymin = 0, ymax = troi1[1][0])
#    ax1.yaxis.tick_right()
    # colorbar1
    divider1 = make_axes_locatable(ax1)
    cax1     = divider1.append_axes("bottom", size="3%", pad = 0.05)
    cbar1    = plt.colorbar(im1, cax=cax1, orientation = 'horizontal')
    cbar1.set_label('')

    ax1.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off')         # ticks along the top edge are off
    

####
    plt.rcParams.update({'font.size': 18})
    fig.set_figheight(8)
    fig.set_figwidth(16)

    plt.savefig('/tmp_14_days/johannes1/maps1.svg', transparent=True)
    plt.savefig('/tmp_14_days/johannes1/maps1.png', transparent=True)
    plt.savefig('/tmp_14_days/johannes1/maps1.eps', transparent=True)
    plt.savefig('/tmp_14_days/johannes1/maps1.pdf', transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
