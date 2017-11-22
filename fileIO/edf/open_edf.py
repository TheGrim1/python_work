from __future__ import print_function

# home: /data/id13/inhouse2/AJ/skript/fileIO/hdf5/open_h5.py

# global imports
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import fabio

# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice


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
