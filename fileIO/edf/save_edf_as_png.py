from __future__ import print_function
from __future__ import absolute_import
# home: /data/id13/inhouse2/AJ/skript/fileIO/hdf5/save_as_jpg.py

import os
import h5py
import sys
import matplotlib.pyplot as plt
import numpy as np

# local imports:

from .open_edf import open_edf
from .plot_edf import plot_edf

def save_as_png(fname, savename):
    
    
        

    data = open_edf(fname)
#    print data.shape

    savefname = os.path.dirname(savename)+ os.path.sep + os.path.basename(savename) + ".png"

    plot_edf(data[:,:],savename = savefname)

    print("saving figure as :\n%s" % savefname)
        


def main(args):

    
    for fname in args:
        
        savename   = os.path.sep.join(["/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/aj_log/compos/images/", os.path.basename(fname)[0:(os.path.basename(fname).find("compo")+5)]])
        save_as_png(fname, savename)

if __name__ == '__main__':
    """ \n1) python <thisfile.py> <arg1> <arg2> etc. 
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
