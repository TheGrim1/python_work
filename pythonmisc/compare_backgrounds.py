from __future__ import print_function

# global imports

import sys, os
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm
import h5py
import numpy as np

# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice

def sum_trois(args):

    ### entering
    fname = args[0]
    frames_per_file = 2000
    
    troi_list = [((1155, 781), (143, 94)), #empty space
                ((99, 100), (89, 96)), # light in top left corner
                ((826, 1173), (70, 45)), # streak towards top left
                ((1210, 1357), (12, 27)), # data top
                ((1247, 1328), (18, 10))] # data left

    troi_names = ['empty space',
                  'light in top left corner',
                  'streak towards top left',
                  'data top',
                  'data left']

    meshshape = [303,85]

    #### dooing
    
    slice_list = []
    for troi in troi_list:
        slice_list.append(troi_to_slice(troi))


    print('opening file %s' % fname)
    f = h5py.File(fname,'r')
    basegroup = 'entry/data/'
    group_list = f[basegroup].keys()
    group_list.sort()
    print('found datasets:')
    print(group_list)

 
    result = np.zeros(shape = (len(troi_list),meshshape[0] * meshshape[1]))

    for group_no, group in enumerate(group_list):
        noframes = f[basegroup+group].shape[0]
        print('='*50)
        print('dataset = %s' %group)
        print('number of frames = %s' % noframes)
        for frame in range(noframes):
            print('dataset %s, frame %s of %s' % (group, frame, noframes))
            for sl_no, sl in enumerate(slice_list):
                data = f[basegroup+group][frame][sl]
                result[sl_no, group_no*frames_per_file + frame] = data.sum()
                #print data.sum()
    f.close()

    result = result.reshape(len(troi_list), meshshape[0], meshshape[1])

    return result
    



if __name__=='__main__':
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
    sum_trois(args)
