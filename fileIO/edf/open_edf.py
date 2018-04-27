from __future__ import print_function

# home: /data/id13/inhouse2/AJ/skript/fileIO/hdf5/open_h5.py

# global imports
import sys, os
import numpy as np
import fabio

# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice


def read_multiframe_edf(source_fname, index_list=None):
    '''
    wrapper to read data[index_list] from a multiframe edf using fabio
    '''

    source_f= fabio.open(source_fname)
    if type(index_list)==type(None):
        index_list = range(source_f.nframes)

    source_f.currentFrame = index_list[0]
    datashape = (len(index_list),source_f.data.shape[0],source_f.data.shape[1])
    datatype = source_f.data.dtype 
    data = np.zeros(shape=(datashape), dtype=datatype)
    for i, source_i in enumerate(index_list):
        #print('reading edf frame {} into data index{}'.format(source_i, i))
        #print('data.sum() = {}'.format(source_f.data.sum()))
        source_f.currentframe = source_i
        data[i] = source_f.data

    return data
    

def open_edf(filename, threshold = 0.0, troi = None):
    'This function opens the specified edf file and returns the frame as a 2D numpy array. Default upper threshold and troi is none.'
    if filename.find(".edf") != -1:   
        
        if troi == None:
            data = fabio.open(filename).data
        else:
            data = fabio.open(filename).data[troi_to_slice(troi)]
#        print troi_to_slice(troi)

        if threshold != 0.0:
            data  = np.where(data < threshold, data, 0)

        return data
    else:
        print("%s is not a .edf file?" %filename)



def main(args):
#  does some plotting which is not suposed to be in this files workflow, but you have to start somewhere
    import matplotlib.pyplot as plt
    for filename in args:
        data = open_edf(filename)
        plt.matshow(data[:,:])
        plt.clim(3000000,8000000)
        plt.show()
    

if __name__ == '__main__':
    
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
