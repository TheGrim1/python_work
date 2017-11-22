
# global imports

import sys, os
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm

# local imports
from open_edf import open_edf
from open_h5 import open_h5

def plot_array(data,
            index    = 0,
            savename = None,
            title    = "Title"):
    'plots the data (nparray) at frame index (if it is 3D). If savename is not None, it saves the plot there. Plotted with title = title.'

## timing
#    print "before plotting1 : %s" %time.time()

    ax1 = plt.subplot(1,1,1,axisbg = (0.9, 0.9, 0.95))
    ax1.figure.set_size_inches(2,7)

    dimension = len(data.shape)
#    print "dimension of data to be plotted is %s" % dimension

    if dimension == 3:
        tobeplotted = data[index,:,:] 
    elif dimension == 2:
        tobeplotted = data[:,:] 
    elif dimension not in (2,3):
        print "invalid data for plotting \ntitle  : %s\n%s" % (title, dimension)
        raise IndexError
## timing
#    print "before plotting2 : %s" %time.time()
    cmin=max(data.min(),0.0001)
    cmax=max(data.max(),0.01)
    ax1.imshow(tobeplotted, norm=LogNorm(vmin = cmin,vmax =cmax), cmap='Greys')

## timing
#    print "after plotting1 : %s" %time.time()

## Labels
    plt.tight_layout()
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.set_yticklabels([])

## draw some lines 

## in the midle of FOV
#    print ax1.get_ylim()
    
#    ax1.plot((20,20), (0,40) ,color = "black", linewidth = 0.5)
## or here:    
#    ax1.plot((13,0), (60,60) ,color = "black", linewidth = 0.5)
#    ax1.plot((13,0), (120,120) ,color = "black", linewidth = 0.5)
#    ax1.plot((13,0), (180,180) ,color = "black", linewidth = 0.5)




    allroi = ((784, 135), (283, 402))
    rroi   = ((929, 341), (33, 55))
    mroi   = ((926, 268), (40, 65))

    ax1 = draw_lines_troi(troi = allroi, axes = ax1)
    ax1 = draw_lines_troi(troi = mroi, color = "blue", axes = ax1)
    ax1 = draw_lines_troi(troi = rroi, color = "red", axes = ax1)

    if savename != None:
        print "saving plot as: \n%s" %savename
        plt.savefig(savename, transparent = True, bbox_incens = "tight")

    plt.title(title)
    plt.show()    

## timing
#    print "after plotting2 : %s" %time.time()

def draw_lines_troi(troi = ((0,0),(10,10)), color = "black", axes = None):
    """
    draw a box into an Eiger sized array plotted into axes = axes around the give troi = troi, with color = color.
    """
    (y,x,dy,dx) = (troi[0][0],troi[0][1],troi[1][0],troi[1][1])
 
    axes.plot((x,x+dx), (y,y) ,color = color, linewidth = 1)
    axes.plot((x,x+dx), (y+dy,y+dy) ,color = color, linewidth = 1)
    axes.plot((x+dx,x+dx), (y,y+dy) ,color = color, linewidth = 1)
    axes.plot((x,x), (y,y+dy) ,color = color, linewidth = 1)
    
    return axes

def plotmany_edf(data):
    'old, use plot_array'
    plt.imshow(data[0,:,:], interpolation = "none")
    plt.clim(0,1000)
    plt.figure(figsize=(10,10))


    title     = "Plotting image no %s of %s" 
    plt.title(title % (0, 0))
    plt.ion()
    
    nimages = data.shape[0]

    for i in range(0,nimages):
        time.sleep(.1)
        plt.title(title % (i, nimages))
        plt.imshow(data[i,:,:])
#        plt.clim(0,10)
        plt.draw()

        

def main(args):
    ' plots and saves edfs or hdf5s, (latter is TODO, only edfs for now)'
    
    framelist = [891,1332]

## timing
#    print "starting time : %s" %time.time()

    for fname in args:
        if fname.endswith('.edf'):
            data = open_edf(fname, threshold = 5000000)
        elif fname.endswith('.h5'):
            data = open_h5(fname, framelist = framelist, threshold = 5000000)
            
## timing
#        print "after open : %s" %time.time()
#        print "data.ndim = %s" %data.ndim
#        print "data.shape = "
 
        print(data.shape)
        if data.ndim !=2:
            for i in range(data.shape[0]):
                savepath     = "/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/aj_log/forjura" 
                savefilename =  os.path.basename(fname)[0:os.path.basename(fname).find("compo")+4] + '_frame%d.png' %framelist[i]
                savename = os.path.normpath(savepath + os.path.sep + savefilename)
                                                                                                                
                plot_array(data, index = i,savename = savename)
        else:
            savepath     = "/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/aj_log/forjura" 
            savefilename =  os.path.basename(fname)[0:os.path.basename(fname).find("compo")+4] + '_rroi.png'
            savename = os.path.normpath(savepath + os.path.sep + savefilename)
            plot_array(data, title = os.path.basename(fname),savename = savename)

## timing
#    print "final time : %s" %time.time() 

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
