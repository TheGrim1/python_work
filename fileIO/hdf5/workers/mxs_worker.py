from __future__ import print_function

import numpy as np
import h5py
import os, sys
import time


sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy, rebin
from fileIO.edf.save_edf import save_edf

def mxs_employer(pickledargs_fname):
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

    
def mxs_data(fname, 
             index = 0,
             framestep = 60,
             data_path = 'entry/data/data',
             verbose = False):
    ### maxprojects and sums over index of an h5 dataset
    ## preserves shape
    ## may profit from buffering of read frames? - TODO bench
    op_starttime = time.time()
    pid             = os.getpid()

    with h5py.File(fname,'r') as source_h5:
        data = source_h5[data_path]
        n = data.shape[index]
        no_sets         = int(n/framestep)    
        newshape        = [data.shape[x] for x in range(len(data.shape)) if x != index]
        max_data        = np.zeros(shape = newshape, dtype=np.int32)
        max_new         = np.zeros_like(max_data)
        sum_data        = np.zeros_like(max_data,np.int64)
        sum_new         = np.zeros_like(max_data,np.int64)


        for i in range(no_sets):
            if verbose:
                print('pid: {} mxs up to frame {}'.format(pid, (i+1)*framestep))
            np.max(data[i*framestep:(i+1)*framestep],axis=index,out=max_new)
            np.max([max_new, max_data],axis=0,out=max_data)
            np.sum(data[i*framestep:(i+1)*framestep],axis=index,out=sum_new)
            sum_data += sum_new
        try:
            if verbose:
                print('pid: {} mxs up to last frame {}'.format(pid, n))        
            np.max(data[no_sets*framestep:],axis=index,out=max_new)
            np.max([max_new, max_data],axis=0,out=max_data)
            np.sum(data[no_sets*framestep:],axis=index,out=sum_new)
            sum_data += sum_new
        except IndexError:
            pass

        op_endtime = time.time()
        op_time = (op_endtime - op_starttime)
        print('='*25)
        print('\ntime taken for mxs of {} frames = {}'.format(n, op_time))
        print(' = {} Hz\n'.format(n/op_time))
        print('='*25) 

    return max_data, sum_data

    
def mxs_worker(pickledargs_fname):
    '''
    copies troi into target_fname[target_datasetpath][target_index] from source_name[source_datasetpath][source_index][troi]
    these dataset have to allready exist with the right shape and dtype 
    no compression, if more than on onf these workers is working on one file! 
    Changes to unpickling here must be updated in h5_scan_nexusversion
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)

    fname = unpickled_args[0]
    data_path = unpickled_args[1]
    axis = unpickled_args[2]
    framestep = unpickled_args[3]
    sum_dest = unpickled_args[4]
    max_dest = unpickled_args[5]
    verbose = unpickled_args[6]
    
    max_data,sum_data =  mxs_data(fname, 
                                  index = axis,
                                  framestep = framestep,
                                  data_path = data_path,
                                  verbose = verbose)
    
    if verbose:
        print('writing files {}'.format(sum_dest, max_dest))
    save_edf(max_data,max_dest)
    save_edf(sum_data,sum_dest)
    
if __name__=='__main__':
    ''' 
    This is used by the lacal function copy_troi_employer(pickledargs_fname),
    DO NOT CHANGE
    '''
    if len(sys.argv)!=2:
        print('usage : python copy_troi_worker <pickled_instruction_list_fname>')
    pickledargs_fname = sys.argv[1]
    mxs_worker(pickledargs_fname)
                        
