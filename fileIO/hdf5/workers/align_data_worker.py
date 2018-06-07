from __future__ import print_function

import numpy as np
import h5py
import os, sys
import scipy.ndimage.measurements as meas
from scipy.ndimage import shift as ndshift

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy


def align_data_employer(pickledargs_fname):
    '''
    gutwrenching way to (almost) completely uncouple the h5 access from the motherprocess
    Can be used to multiprocess.pool control the workers. Allows another layer of parallelisation :D
    '''
    fname =  __file__
    cmd = 'python {} {}'.format(fname, pickledargs_fname)
    
    os_response = os.system(cmd)
    if os_response >1:
        raise ValueError('in {}\nos.system() has responded with errorcode {} in process {}'.format(fname, os_response, os.getpid))

    
def align_data_worker(pickledargs_fname):
    '''
    interates align routine over source_name[source_grouppath][dataset_path] and stores results in target_fname[target_group][results_path]
    these datasets are created here
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)

    print(os.getpid()) 
    print(unpickled_args)
   
    target_fname = unpickled_args[0]
    target_grouppath = unpickled_args[1]
    source_fname = unpickled_args[2]
    source_grouppath = unpickled_args[3]
    Theta = unpickled_args[4]
    mapshape = unpickled_args[5]
    shift = unpickled_args[6]
    lines_shift = unpickled_args[7]
    verbose = unpickled_args[8]
        
    if verbose:
        print('='*25)
        print('process {} is doing'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)

    with h5py.File(target_fname) as target_file:
        target_group = target_file.create_group(target_grouppath)
        
        with h5py.File(source_fname,'r') as source_file:
            
            source_group = source_file[source_grouppath]
            dtype = source_group['tth_radial/I'].dtype
            data =  np.asarray(source_group['tth_2D/data'])
            datashape = data.shape
            data = data.reshape(list(mapshape)+list(datashape[1:]))

            if type(lines_shift)!=type(None):
                for i,map_lines in enumerate(data):
                    if lines_shift[i]!=0:
                        ndshift(map_lines, lines_shift[i]+[0]*(data.ndim-1),output=data[i])
            
            data_sum = np.asarray(source_group['tth_radial/I']).sum(1).reshape(mapshape)
            ndshift(data_sum, shift, output = data_sum)
            data_max = np.asarray(source_group['tth_radial/I']).max(1).reshape(mapshape)
            ndshift(data_max, shift, output = data_max)
                        
            target_group.create_dataset('sum', data=data_sum)
            target_group.create_dataset('max', data=data_max)

            ndshift(data, shift=list(shift)+[0]*(data.ndim-2), output=data)
            target_group.create_dataset('data', data=data)
        
        target_file.flush()
            
    if verbose:
        print('process {} is done'.format(os.getpid()))
        print('='*25)


if __name__=='__main__':
    ''' 
    This is used by the local function integrate_troi_employer(pickledargs_fname),
    DO NOT CHANGE
    '''
    if len(sys.argv)!=2:
        print('usage : python integrate_troi_worker <pickled_instruction_list_fname>')
    pickledargs_fname = sys.argv[1]
    align_data_worker(pickledargs_fname)
                        
