from __future__ import print_function

# home: /data/id13/inhouse2/AJ/skript/fileIO/hdf5/open_h5.py

# global imports
import h5py
import sys, os
import matplotlib.pyplot as plt
import time

# local imports
from open_h5 import open_h5

def plot_h5(data,
            index=0, 
            title = "Title"):
    dimension = len(data.shape)
    print("dimension of data to be plotted is %s" % dimension)
    if dimension == 3:
        plt.imshow(data[index,:,:], interpolation = 'none')
    elif dimension == 2:
        plt.imshow(data[:,:], interpolation = 'none')
    elif dimension not in (2,3):
        print("invalid data for plotting \ntitle  : %s\n%s" % (title, dimension))
#    plt.clim(0,0.001)

    plt.show()

    plt.title(title)


def plotmany_h5(data,index):


    ax1 = plt.subplot(1,1,1,axisbg = (0.9, 0.9, 0.95))
    ax1.figure.set_size_inches(10,10)

    title     = "Plotting image no %s of %s" 
    ax1.title(title % (0, 0))
    ax1.ion()

    if dimension == 3:
        toplot = data[index,:,:]
    elif dimension == 2:
        toplot = data[:,:]
    elif dimension not in (2,3):
        print("invalid data for plotting \ntitle  : %s\n%s" % (title, dimension))

    ax1.pcolor(toplot,norm=LogNorm(vmin=max(data.min(),0.0001),vmax=max(data.max(),0.01)), cmap='PuBu')
    nimages = data.shape[0]

        
    plt.show()
        

def main(args):
    'also does the plotting'
    
    for fname in args:
        data = open_h5(fname, framelist = [529,530,532], threshold = 5000000)
#        print 'the data shape is:' 
#        print data.shape
        if data.ndim != 2:
            plotmany_h5(data)
        else:
            plot_h5(data, title = os.path.basename(fname))

    

if __name__ == '__main__':
  
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc. 
\n2) python <thisfile.py> -f <file containing args as lines> 
\n3) find <*yoursearch* -> arg1 etc.> | python <thisfile.py> 
"""
  
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
