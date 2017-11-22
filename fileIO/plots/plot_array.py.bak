
# global imports

import sys, os
import matplotlib.pyplot as plt
plt.ion()
import time
from matplotlib.colors import LogNorm
import numpy as np

# local imports
# local imports
path_list = os.path.dirname(__file__).split(os.path.sep)
importpath_list = []
if 'skript' in path_list:
    for folder in path_list:
        importpath_list.append(folder)
        if folder == 'skript':
            break
importpath = os.path.sep.join(importpath_list)
sys.path.append(importpath)        
import fileIO.plots.plot_tools as pt



def plot_array(data,
               index     = None,
               savename  = None,
               perc_low  = 0,
               perc_high = 100,
               title     = 'title'):
    'plots the data (nparray) at frame index (if it is 3D). If savename is not None, it saves the plot using save_plot(plt,savename). Plotted with title = title.'


## timing
#    print "before plotting1 : %s" %time.time()


    dimension = len(data.shape)
#    print "dimension of data to be plotted is %s" % dimension

    if dimension == 3:
        if index == None:
            index_list = range(data.shape[0])
        elif len(index)==1:
            dimension == 2
            data = data[index,:,:]
        else:
            index_list = index
    elif dimension not in (2,3):
        print "invalid data for plotting \ntitle  : %s\n%s" % (title, dimension)
        raise IndexError
## timing
#    print "before plotting2 : %s" %time.time()

    if dimension == 2:
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1,axisbg = (0.9, 0.9, 0.95))
        ax1.figure.set_size_inches(10,10)
        (vmin,vmax) = pt.get_vcolor(data, low=perc_low, high=perc_high)
        ax1.matshow(data,
                    vmin=vmin,
                    vmax=vmax)

    elif dimension == 3:
        nframes = len(index_list)
        (ncols, nrows) = pt.factorize(nframes)
        
        
        fig, axes = plt.subplots(nrows=np.int(nrows),
                                 ncols=np.int(ncols))
        axes_flat = axes.flatten()
        
        for index_no, i in enumerate(index_list):
            tobeplotted = data[i,:,:]
            (vmin,vmax) = pt.get_vcolor(tobeplotted, low=perc_low, high=perc_high)

            axes_flat[index_no].matshow(tobeplotted,
                                        vmin=vmin,
                                        vmax=vmax)
            axes_flat[index_no].tick_params(direction = 'out', labelbottom = True, labeltop = False)
            if type(title) == list:
                axes_flat[index_no].set_title(title[index_no])
            else:
                axes_flat[index_no].set_title(title)
                
## ticks## ticks ## ticks ## ticks ## ticks ## ticks ## ticks ## ticks 



    if savename != None:
        plt.savefig(savename,transparent = True, bbox_incens = "tight", dpi =300)

    #    plt.title(title)
    print 'plotting array:' 
    plt.show()    
    print 'plot_array finished'



def main(args):
    ' plots and saves edfs or hdf5s, (latter is TODO, only edfs for now)'

    sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
    from fileIO.edf.open_edf import open_edf
    from fileIO.hdf5.open_h5 import open_h5

## optional restrictions
    
#    framelist = [891,1332]
#    framelist = [511]
#    allroi = ((784, 135), (283, 402))

## timing
#    print "starting time : %s" %time.time()

    for fname in args:
        if fname.endswith('.edf'):
            data = open_edf(fname, threshold = 5000000)
        elif fname.endswith('.h5'):
            data = open_h5(fname, framelist = framelist, threshold = 5000000,troi = allroi)
            
## timing
#        print "after open : %s" %time.time()
#        print "data.ndim = %s" %data.ndim
#        print "data.shape = "
 
        print(data.shape)
        if data.ndim !=2:
            for i in range(data.shape[0]):
#                savepath     = "/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/aj_log/forjura" 
#                savefilename =  os.path.basename(fname)[0:os.path.basename(fname).find("data")] + '_frame%d.png' %framelist[i]
#                savename = os.path.normpath(savepath + os.path.sep + savefilename)
                savename = None
                plot_array(data, index = i,savename = savename)
        else:
#            savepath     = "/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/aj_log/forjura" 
#            savefilename =  os.path.basename(fname)[0:os.path.basename(fname).find("data")] + '_frame%d.png' %framelist[0]           

#            savename = os.path.normpath(savepath + os.path.sep + savefilename)
            savename = None
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
