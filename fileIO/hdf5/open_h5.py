
# home: /data/id13/inhouse2/AJ/skript/fileIO/hdf5/open_h5.py

import h5py
import os,sys
import numpy as np

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice

def open_h5(fname,framelist=None,group="entry/data/data", threshold = 0, troi = None, verbose = False):
    'This function opens the specified hdf5 file at default group = entry/data/data and returns the frames asa 3D numpy array. Default framelist is None (gives all frames), default threshold is none'

    if fname.find(".h5") != -1:
        f       = h5py.File(fname, "r")
        if verbose:
            print 'found shape = '
            print f[group].shape
        if framelist == None:
            framelist = slice(0,f[group].shape[0],1)

        else:
            if len(framelist) == 1:
                framelist = slice(framelist[0],framelist[0]+1,1)                
        if troi == None:
            troi = ((0,0),(f[group].shape[1],f[group].shape[2]))

            if verbose:
                print('reading troi:')
                print(troi)
            
        try:
            data = f[group][framelist][:,troi_to_slice(troi)[0],troi_to_slice(troi)[1]]
            
            if threshold >= 1.0:
                data  = np.where(data < threshold, data, 0)
            return data
        except KeyError:
            print "did not find %s in %s" % (group, fname)
    else:
        print "%s is not a .h5 file" %fname



            
def open_h5_old(fname,framelist=None,group="entry/data/data", threshold = 0, troi = None):
    '''This function opens the specified hdf5 file at default group = entry/data/data and returns the frames asa 3D numpy array. Default framelist is None (gives all frames), default threshold is none
    * <2017-01-11 Wed> -> _old
    There seems to be a different order to the indeces, so I get a problem with h5_scan.read_data ?!
    changed :      
    data     = f[group][framelist][troi_to_slice(troi)]       
    to:
    data = f[group][framelist][:,troi_to_slice(troi)[0],troi_to_slice(troi)[1]]

    '''

    if fname.find(".h5") != -1:
        f       = h5py.File(fname, "r")
        print 'found shape = '
        print f[group].shape
        if framelist == None:
            framelist = slice(0,f[group].shape[0],1)

        if troi == None:
            troi = ((0,0),(f[group].shape[1],f[group].shape[2]))
#        (ystart,yend,xstart,xend) = troi_to_slice(troi)
            
        try:
            data     = f[group][framelist][troi_to_slice(troi)]

            if threshold >= 1.0:
                data  = np.where(data < threshold, data, 0)
            return data
        except KeyError:
            print "did not find %s in %s" % (group, fname)
    else:
        print "%s is not a .h5 file" %fname





def main(args):
#  does some plotting which is not suposed to be in this files workflow, but you have to start somewhere
    
    for fname in args:
        data = open_h5(fname,framelist=[1],threshold = 100)
        plt.imshow(data[0,:,:])
        plt.clim(0,1)
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
