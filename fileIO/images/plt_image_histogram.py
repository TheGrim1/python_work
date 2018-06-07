
from __future__ import print_function
from __future__ import division

import sys, os
import fabio

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/data/id13/inhouse2/AJ/skript')

import fileIO.images.image_tools as it

def main(args):    
    
    mask_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-06_inh_ihsc1547_mro/PROCESS/aj_log/mask.edf'

    if 'mask_fname'in dir():        
        mask = fabio.open(mask_fname).data
        not_mask = np.where(mask,0,1)
    else:
        mask=None

    no_bins = 10
    
    for fname in args:
        if fname.find('.edf')>0:
            data=fabio.open(fname).data

        if type(mask)!= type(None):
            flat_data = data[np.where(not_mask)].flatten()
        else:
            flat_data = data.flatten()

        low = np.percentile(flat_data,1)
        high = np.percentile(flat_data,99)
        bins = np.linspace(low, high, 11)
        bincenters = (bins[:-1]+bins[1:])/2
        hist = np.histogram(flat_data, bins=bins)
        plt.bar(bincenters,hist[0],width = bins[0]-bins[1])
        plt.show()


        

if __name__ == '__main__':
    
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc.  \n2)
python <thisfile.py> -f <file containing args as lines> \n3) find
<*yoursearch* -> arg1 etc.> | python <thisfile.py> """

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
    
    main(args)

    
