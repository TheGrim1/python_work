from __future__ import print_function

# global imports
from multiprocessing import Pool
import sys, os
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LogNorm
import h5py
import numpy as np
import json

# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice


def task(inargs):
               
    fname      = inargs[0]
    group      = 'entry/data/' + inargs[1]
    slice_list = inargs[2]
    
    print("doing: group %s in process %s" % (group,os.getpid()))
    f = h5py.File(fname,'r')

    noframes = f[group].shape[0]
    print('number of frames = %s' % noframes)
    
    result = np.zeros(shape = (len(slice_list),noframes))
    
    for frame in range(noframes):
        print('dataset %s, frame %s of %s' % (group, frame, noframes))
        for sl_no, sl in enumerate(slice_list):
            data = f[group][frame][sl]
            result[sl_no, frame] = data.sum()
            #print data.sum()
    f.close()

    return result

def sum_trois(args):

    ### entering
    fname = args[0]
    frames_per_file = 2000
    
    troi_list = [((1155, 781), (143, 94)), #empty space left
                ((406, 1416), (100, 134)),# empty space top
                ((99, 100), (89, 96)), # light in top left corner
                ((826, 1173), (70, 45)), # streak towards top left
                ((1692, 1144), (15, 19)),# streak towards bottom left
                ((1247, 1363), (10, 11)), # behind BS
                ((1210, 1357), (12, 27)), # data top
                ((1247, 1328), (18, 10))] # data left

    troi_names = ['empty space left',
                  'empty space top',
                  'light in top left corner',
                  'streak towards top left',
                  'streak towards bottom left',
                  'behind BS',
                  'data top',
                  'data left']

    meshshape = [101,255]

    #### finding data sizes
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
    f.close()


    #### setting up parrallel processing:
    
    todolist = []
    for group_no, group in enumerate(group_list):
        todolist.append([fname,
                         group,
                         slice_list])

    noprocesses = len(group_list)
    print('Creating pool with %d processes\n' % noprocesses)
        
    #for item in todolist:
    #    for subitem in item:
    #        print subitem
            
    pool = Pool(processes=noprocesses)


    #### dooing
    result = pool.map(task,todolist)

    result = np.hstack(result)
    
    # f.close()

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
