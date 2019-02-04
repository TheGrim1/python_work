from __future__ import print_function

import numpy as np
import h5py
import os, sys

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
from fileIO.edf.save_edf import save_edf

def bliss_mxs_employer(pickledargs_fname):
    '''
    gutwrenching way to completely uncouple the h5 access from the motherprocess
    Can be used to multiprocess.pool control the workers.
    '''
    fname =  __file__
    cmd = 'python {} {}'.format(fname, pickledargs_fname)

    # import inspect
    # linno = inspect.currentframe().f_lineno
    # print('DEBUG:\nin ' + __file__ + '\nline '+str(linno))
    # print(cmd)

    os_response = os.system(cmd)
    if os_response >1:
        raise ValueError('in {}\nos.system() has responded with errorcode {} in process {}'.format(fname, os_response, os.getpid))

def bliss_mxs_worker(pickledargs_fname):
    '''
    copies troi into target_fname[target_datasetpath][target_index] from source_name[source_datasetpath][source_index][troi]
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)
    # import inspect

    print(unpickled_args)

    source_fname = unpickled_args[0]
    target_path = unpickled_args[1]
    verbose = unpickled_args[2]
    
    if verbose:
        print('='*25)
        print('process {} is summing'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)
        print('='*25)



    raw_data_path = 'entry_0000/instrument/E-08-0106/image_data'
    pid = os.getpid()
    with h5py.File(source_fname,'r') as source_h5:
        
        data = source_h5[raw_data_path]
        data_len = data.shape[0]
        no_slices = data_len/60
        index_bounds = [int(x) for x in np.linspace(0,data_len,no_slices)]
        slices_list = [slice(x0,x1) for x0,x1 in zip(index_bounds[:-1],index_bounds[1:])]

        data_max = np.zeros_like(data[0])
        data_sum = np.zeros_like(data[0], dtype=np.uint64)

        for i, curr_slice in enumerate(slices_list):
            print('{}, slice {} of {}'.format(pid,i,no_slices))
            subset = np.asarray(data[curr_slice])
            curr_max = subset.max(axis=0)
            curr_sum = subset.sum(axis=0)
            data_max = np.where(data_max>curr_max, data_max, curr_max)
            data_sum += curr_sum

        fname = os.path.splitext(os.path.basename(source_fname))[0]
        save_edf(filename=target_path+'/{}_{}_sum.edf'.format(fname,pid), data=data_sum)
        save_edf(filename=target_path+'{}_{}_max.edf'.format(fname,pid), data=data_max)
                    

    if verbose:
        print('process {} is done'.format(os.getpid()))
        print('='*25)


        
if __name__=='__main__':
    ''' 
    This is used by the lacal function qxyz_regroup_employer(pickledargs_fname),
    DO NOT CHANGE
    '''
    if len(sys.argv)!=2:
        print('usage : python qxyz_regroup_worker <pickled_instruction_list_fname>')
    pickledargs_fname = sys.argv[1]
    bliss_mxs_worker(pickledargs_fname)
                        
