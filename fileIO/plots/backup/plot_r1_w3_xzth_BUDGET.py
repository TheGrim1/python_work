from __future__ import print_function
from __future__ import division

# global imports

from past.utils import old_div
import sys, os
import matplotlib.pyplot as plt
import time
import matplotlib
from matplotlib.colors import LogNorm
import numpy as np

# local imports
# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.edf.open_edf import open_edf
from fileIO.hdf5.open_h5 import open_h5

def plot_array(data,
            index    = 0,
            savename = None,
            title    = "Title"):
    'plots the data (nparray) at frame index (if it is 3D). If savename is not None, it saves the plot using save_plot(plt,savename). Plotted with title = title.'
    matplotlib.rc('font', family='STIXGeneral', size = 16)


## timing
#    print "before plotting1 : %s" %time.time()
## size## size## size## size## size## size## size## size## size## size## size## 
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1,axisbg = (0.9, 0.9, 0.95))
    ax2 = fig.add_subplot(2,1,2,axisbg = (0.9, 0.9, 0.95))
    ax1.figure.set_size_inches(5,4.5)
#    ax2.figure.set_size_inches(3,10)

 
    c1min=max(data[0,:,:].min(),0.00001)
    c1max=max(data[0,:,:].max(),10)
#    ax1.imshow(tobeplotted, norm=LogNorm(vmin = cmin,vmax =cmax), cmap='hot')
    ax1.imshow(data[0,:,:],vmin = c1min,vmax =c1max, cmap='hot')

    c2min=max(data[1,:,:].min(),0.00001)
    c2max=max(data[1,:,:].max(),10)
#    ax1.imshow(tobeplotted, norm=LogNorm(vmin = cmin,vmax =cmax), cmap='hot')
    ax2.imshow(data[1,:,:],vmin = c2min,vmax =c2max, cmap='hot')





## timing
#    print "after plotting1 : %s" %time.time()

## Labels## Labels## Labels## Labels## Labels## Labels## Labels## Labels

    ylabelpertick = old_div(10.,60)
    xlabelpertick = old_div(32.,165)
    ylabels       = np.arange(0,12,2)
    xlabels       = np.arange(0,35,5)
    yticks        = old_div(ylabels,ylabelpertick)
    xticks        = old_div(xlabels,xlabelpertick)

#    plt.tight_layout()
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([])
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels)


    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabels)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels)


## ticks## ticks ## ticks ## ticks ## ticks ## ticks ## ticks ## ticks 
    ax1.tick_params(direction = 'out')

    ax2.tick_params(direction = 'out')

    
## tilte and axixlabels ## tilte and axixlabels ## tilte and axixlabels ## 
    ax1.set_ylabel('z [$\mu$m]')
#    ax1.set_xlabel('x [$\mu$m]')
    angle = old_div(float(savename[savename.find('xzth__')+6:savename.find('xzth__')+9]),10)

    ax1.set_title('rocking angle $\omega$ = {:2.1f}'.format(angle))
    ax2.set_ylabel('z [$\mu$m]')
    ax2.set_xlabel('x [$\mu$m]')
#    ax2.set_title('ROI 2')

    ax1.text(5,10,"ROI 1",color = 'white')
    ax2.text(5,10,"ROI 2",color = 'white')

#    plt.title(title)    

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
        plt.savefig(savename,transparent = True, dpi =200)




#    plt.show()    
    ax1.clear()
    ax2.clear()
    fig.clf()


## timing
#    print "after plotting2 : %s" %time.time()

def draw_lines_troi(troi = ((0,0),(10,10)), color = "black", axes = None, fov = (0,0)):
    """
    draw a box into an Eiger sized array plotted into axes = axes around the give troi = troi, with color = color.
    """
    (y,x,dy,dx) = (troi[0][0]-fov[0],troi[0][1]-fov[1],troi[1][0],troi[1][1])
 
    axes.plot((x,x+dx), (y,y) ,color = color, linewidth = 2)
    axes.plot((x,x+dx), (y+dy,y+dy) ,color = color, linewidth = 2)
    axes.plot((x+dx,x+dx), (y,y+dy) ,color = color, linewidth = 2)
    axes.plot((x,x), (y,y+dy) ,color = color, linewidth = 2)
    
    return axes

def main(args):
    ' plots and saves edfs or hdf5s, (latter is TODO, only edfs for now)'

## optional restrictions
    
#    framelist = [891,1332]
#    framelist = [511]
#    allroi = ((784, 135), (283, 402))

## timing
#    print "starting time : %s" %time.time()

# opening data
    for (i,fname) in enumerate(args):
        if fname.endswith('.edf'):
            if fname.find('ROI1')!=-1 and args[i+1].find('ROI2')!=-1:
                data    = open_edf(args[i], threshold = 5000000)
                toplot  = np.zeros(shape=(2,data.shape[0],data.shape[1]))
                toplot[0,:,:]=data
                data    = open_edf(args[i+1], threshold = 5000000)
                toplot[1,:,:]=data
                                   
            

           
## timing
#        print "after open : %s" %time.time()
#        print "data.ndim = %s" %data.ndim
#        print "data.shape = "
 
                print(toplot.shape)
                print(fname)


# assigning savename
                savepath     = "/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/compos/r1_w3_xzth/png/"
                savefilename =  os.path.basename(fname)[0:os.path.basename(fname).find(".edf")]
                savename     = os.path.normpath(savepath + os.path.sep + savefilename)

#           savename = None

# plotting

                plot_array(toplot, title = os.path.basename(fname),savename = savename)

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
