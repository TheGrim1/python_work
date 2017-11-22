from __future__ import print_function

# global imports

from builtins import range
import sys, os
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm
import numpy as np

# local imports
# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.edf.open_edf import open_edf
from fileIO.hdf5.open_h5 import open_h5



def plot_rgb(data,
            savename = None,
            title    = "Title"):
    'plots the data (nparray) as rgb. If savename is not None, it saves the plot as savename.png. Plotted with title = title.'



## timing
#    print "before plotting1 : %s" %time.time()
## size## size## size## size## size## size## size## size## size## size## size## 
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1,axisbg = (0.9, 0.9, 0.95))
    ax1.figure.set_size_inches(2,5)

    dimension = len(data.shape)
#    print "dimension of data to be plotted is %s" % dimension
    
    tobeplotted = data 
    if dimension not in (3,4):
        print("invalid data for plotting \ntitle  : %s\ndimension : %s" % (title, dimension))
        raise IndexError
## timing
#    print "before plotting2 : %s" %time.time()

## color ## color ## color ## color ## color ## color ## color ## color ## color 
    ax1.imshow(tobeplotted)

## timing
#    print "after plotting1 : %s" %time.time()

## Labels## Labels## Labels## Labels## Labels## Labels## Labels## Labels
    plt.tight_layout()
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_yticklabels([])

## draw some lines ## draw some lines ## draw some lines ## draw some lines 

## in the midle of FOV
#    print ax1.get_ylim()
    
#    ax1.plot((20,20), (0,40) ,color = "black", linewidth = 0.5)
## or here:    
    # ax1.plot((13,0), (60,60) ,color = "black", linewidth = 0.5)
    # ax1.plot((13,0), (120,120) ,color = "black", linewidth = 0.5)
    # ax1.plot((13,0), (180,180) ,color = "black", linewidth = 0.5)

#    rroi = ((928,293),(37,44))	
#    m2roi = ((928,339),(37,16))
#    m1roi   = ((928,355), (37, 55))
#    mroi   = ((926, 268), (40, 65))

#    ax1 = draw_lines_troi(troi = m1roi, color = "blue", axes = ax1, fov =(784,135))
#    ax1 = draw_lines_troi(troi = m2roi, color = "green", axes = ax1, fov =(784,135))
#    ax1 = draw_lines_troi(troi = rroi, color = "red", axes = ax1, fov =(784,135))

    if savename != None:
        print("saving plot as: \n%s" %savename)
        plt.savefig(savename, transparent = True, bbox_incens = "tight")

#    plt.title(title)

    plt.show()    
    ax1.clear()
    fig.clf()


def main(args):
    'plot 3 edf files as one rgb, optional savename.png'
    
    if len(args) != 3:
        print("please specify 3 files and not %s" %len(args))
        
    i           = 0
    r           = open_edf(args[0])
#    print 'red shape = ' 
#    print r.shape
    data        = np.zeros(shape=(r.shape[0],r.shape[1],3))
    data[:,:,0] = r


    for i in range(len(args)):
        data[:,:,i] = open_edf(args[i])

    plot_rgb(data)

if __name__ == '__main__':
    
    args = []
    if len(sys.argv) > 1:
        if sys.argv[1].find("-f")!= -1:
            f = open(sys.argv[2]) 
            for line in f:
                args.append(line.rstrip())
        else:
            args=sys.argv[1:]
    else:
        f = sys.stdin
        for line in f:
            args.append(line.rstrip())
    
#    print args
    main(args)
