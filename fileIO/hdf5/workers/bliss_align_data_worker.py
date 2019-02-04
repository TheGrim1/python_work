from __future__ import print_function

import numpy as np
import h5py
import os, sys
import scipy.ndimage.measurements as meas
from scipy.ndimage import shift as ndshift

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy
from simplecalc.image_deglitch import data_stack_shift


def bliss_align_data_employer(pickledargs_fname):
    '''
    gutwrenching way to (almost) completely uncouple the h5 access from the motherprocess
    Can be used to multiprocess.pool control the workers. Allows another layer of parallelisation :D
    '''
    fname =  __file__
    cmd = 'python {} {}'.format(fname, pickledargs_fname)
    
    os_response = os.system(cmd)
    if os_response >1:
        raise ValueError('in {}\nos.system() has responded with errorcode {} in process {}'.format(fname, os_response, os.getpid))

    
def bliss_align_data_worker(pickledargs_fname):
    '''
    interates align routine over source_name[source_grouppath][dataset_path] and stores results in target_fname[target_group][results_path]
    these datasets are created here
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)

    print('pid: {} working on {}'.format(os.getpid(), unpickled_args[0]))

    target_fname = unpickled_args[0]
    source_fname = unpickled_args[1]
    source_grouppath = unpickled_args[2]
    troi = unpickled_args[3]
    mask_fname = unpickled_args[4]
    phi_pos = float(unpickled_args[5])
    map_shape = unpickled_args[6]
    shift = unpickled_args[7]
    verbose = unpickled_args[8]

    if verbose:
        print('='*25)
        print('process {} is doing'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)

    if mask_fname != None:
        import fabio
        mask = np.asarray(fabio.open(mask_fname).data)[troi_to_slice(troi)]
        apply_mask = True
    else:
        apply_mask = False


    with h5py.File(target_fname) as target_file:        
        with h5py.File(source_fname,'r') as source_file:

            target_group = target_file.create_group('shifted_data')
            source_ds = source_file[source_grouppath]['data']
            dtype = source_ds.dtype
            if apply_mask:
                data_ds =  source_ds[tuple([slice(None,None,None)]+list(troi_to_slice(troi)))]*mask
            else:
                data_ds =  source_ds[tuple([slice(None,None,None)]+list(troi_to_slice(troi)))]
            # cast to uint64 to avoid rounding issues and improve compression
            data = np.asarray(data_ds*1000, dtype = np.uint64)
            datashape = data.shape
            data = data.reshape(list(map_shape)+list(datashape[1:]))

            data = data_stack_shift(data, -1*shift, None)

            target_group.create_dataset('data', data=data, compression='lzf')
            target_group.create_dataset('max', data=data.max(axis=2).max(axis=2), compression='lzf')
            target_group.create_dataset('sum', data=data.sum(axis=2).sum(axis=2), compression='lzf')
            target_group.create_dataset('phi', data=phi_pos)

        
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
    bliss_align_data_worker(pickledargs_fname)
                        
