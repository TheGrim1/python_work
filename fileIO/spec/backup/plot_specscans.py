from __future__ import print_function
import sys, os
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt


# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.spec.open_scan import open_scan
from simplecalc.image_align import image_align



def main(args):
    'plots a range of spec scans next to each other'

    fname    = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/DATA/AJ2c_after/AJ2c_after.dat'
    scanlist = list(range(76,92))

    allstacked = open_scan(fname = fname, scanlist = scanlist, counter = 'ball01')

# test some slight filtering to even things out doesnt change much
#    for i in range(len(allstacked[0,0,:])):
#        nd.filters.gaussian_filter(allstacked[:,:,i], sigma = 2, output = allstacked[:,:,i])

#    (allstacked, shift) = sift_align(np.float32(allstacked))
#    mode = {'mode':'crosscorrelation','alignment':(-1,0)}
    mode = {'mode':'mask','alignment':(-1,0),'threshold':np.percentile(allstacked,60)}


    (allstacked, shift) = image_align(np.float32(allstacked),mode)

    data       = np.reshape(allstacked.T,(allstacked.shape[1]*allstacked.shape[2],allstacked.shape[0]))


    # print 'data.shape ='
    # print data.shape

    print('shift found with %s alignment = ' % mode['mode'])
    print(shift)

    plt.imshow(data.T, vmin = data.min(), vmax= data.max())

    plt.show()

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
