from __future__ import print_function
from __future__ import absolute_import
# home: /data/id13/inhouse2/AJ/skript/fileIO/hdf5/save_as_jpg.py

from builtins import range
import os
import h5py
import sys
import matplotlib.pyplot as plt
import numpy as np

# local imports:

from .open_h5 import open_h5_ROI
from plot_h5_SSDPoster import plot_h5

def save_as_png(fname, savename, framelist, ROI  = ((1222,470),(40,40))):
    
    
              
    data = open_h5_ROI(fname, framelist,ROI = ROI, threshold = 65000)
#    print data.shape

    for i in range(len(framelist)):

        savefname = os.path.dirname(savename)+"/frame%s_ROI%s_" % (framelist[i] , ROI[0][0]) + os.path.basename(savename) + ".png"

        plot_h5(data[i,:,:],savename = savefname)

        print("saving figure as :\n%s" % savefname)



def main(args):

    
    for fname in args:
        
        savename   = os.path.sep.join(["/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/PROCESS/SESSION22/images", os.path.basename(fname)[0:(os.path.basename(fname).find("data")+4)]])
        framelist  = [60,81,82,83,84,121,158,159,160,161,182]
        ROI        = ((1222,470),(40,40))
        save_as_png(fname, savename, framelist, ROI)

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
