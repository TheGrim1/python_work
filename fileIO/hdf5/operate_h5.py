from __future__ import print_function
from __future__ import absolute_import


USAGE = """ \n1) python <thisfile.py> <OPTIONS ><arg1> <arg2> etc. 
\n2) find <*yoursearch* -> arg1 etc.> | python <OPTIONS> <thisfile.py> 
\n--------
\n operates on each h5 (datafile), ouputs a file each
\n<OPTIONS>:
\n    max - max-proj
\n    sum - sum
\n    mxs - max and sum
"""

### average over an h5 datasets first index, saves each dataset that was averaged as avg_data.h5
# backup:
# cp avg_h5.py /data/id13/inhouse2/AJ/skript/fileIO/hdf5/avg_h5.py

# global imports
import h5py
import sys
import os
import numpy as np
from multiprocessing import Pool
import time

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
# local
from fileIO.hdf5.save_h5 import save_h5
from fileIO.hdf5.open_h5 import open_h5
from fileIO.hdf5.frame_getter import data_getter

# from plot_h5 import plot_h5
from pythonmisc.worker_suicide import worker_init
def sum_data(fname,
             index = 0,
             framestep = 10,
             verbose = True): 
    ### sumsa over index of an h5 dataset
    ## preserves shape
    op_starttime = time.time()
    pid             = os.getpid()
    data            = data_getter(fname)
    n               = data.shape[index]
    
    
    no_sets         = int(n/framestep)
    newshape        = [data.shape[x] for x in range(len(data.shape)) if x != index]
    sum_data         = np.zeros(shape = newshape, dtype=np.int32)
    sum_new         = np.zeros_like(sum_data)
    
    for i in range(no_sets):
        if verbose:
            print('pid: {} sum up to frame {}'.format(pid, (i+1)*framestep))
        np.sum(data[i*framestep:(i+1)*framestep],axis=index,out=sum_new)
        sum_data += sum_new
    try:
        if verbose:
            print('pid: {} sum up to last frame {}'.format(pid, n)) 
        np.sum(data[no_sets*framestep:],axis=index,out=sum_new)
        sum_data += sum_new
            
    except IndexError:
        pass

    op_endtime = time.time()
    op_time = (op_endtime - op_starttime)
    print('='*25)
    print('\ntime taken for sum of {} frames = {}'.format(n, op_time))
    print(' = {} Hz\n'.format(n/op_time))
    print('='*25) 
    
    return sum_data


def max_data(fname, 
             index = 0,
             framestep = 10,
             verbose = True):
    ### maxprojects over index of an h5 dataset
    ## preserves shape
    op_starttime = time.time()
    data            = data_getter(fname,verbose=True)
    pid             = os.getpid()
    n               = data.shape[index]
    no_sets         = int(n/framestep)
    
    op_starttime = time.time()
    
    if verbose:
        print('frames found {}'.format(n))
    
    newshape        = [data.shape[x] for x in range(len(data.shape)) if x != index]
    max_data        = np.zeros(shape = newshape, dtype=np.int32)
    max_new         = np.zeros_like(max_data)

    for i in range(no_sets):
        if verbose:
            print('pid: {} max up to frame {}'.format(pid, (i+1)*framestep))
        np.max(data[i*framestep:(i+1)*framestep],axis=index,out=max_new)
        np.max([max_new, max_data],axis=0,out=max_data)
    try:
        if verbose:
            print('pid: {} max up to last frame {}'.format(pid, n))       
        np.max(data[no_sets*framestep:],axis=index,out=max_new)
        np.max([max_new, max_data],axis=0,out=max_data)
    except IndexError:
        pass
    op_endtime = time.time()
    op_time = (op_endtime - op_starttime)
    print('='*25)
    print('\ntime taken for max of {} frames = {}'.format(n, op_time))
    print(' = {} Hz\n'.format(n/op_time))
    print('='*25) 
        
    return max_data



def mxs_data(fname, 
             index = 0,
             framestep = 60,
             verbose = True):
    ### maxprojects and sums over index of an h5 dataset
    ## preserves shape
    ## may profit from buffering of read frames? - TODO bench
    op_starttime = time.time()
    pid             = os.getpid()
    data            = data_getter(fname)
    n               = data.shape[index]
    no_sets         = int(n/framestep)    
    newshape        = [data.shape[x] for x in range(len(data.shape)) if x != index]
    max_data        = np.zeros(shape = newshape, dtype=np.int32)
    max_new         = np.zeros_like(max_data)
    sum_data        = np.zeros_like(max_data)
    sum_new         = np.zeros_like(max_data)
    
    
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


def operation_worker(args):
    option, fname,rel_dest_path, verbose = args

    dest_path = os.path.realpath((os.path.dirname(fname)+rel_dest_path))
    
    if not os.path.exists(dest_path):
        print('path not found')
        print('path {}\n rel_dest_path {}\nfname {}'.format(dest_path, rel_dest_path, fname))
        sys.exit(0)

    if verbose:
        print('pid: {} running {} on {}'.format(os.getpid(),option, fname))
        
    if option=='max':
        data = max_data(fname,verbose=verbose)
        new_fname = os.path.sep.join([dest_path,option,option+'_'+os.path.basename(fname)]) 
        save_h5(data, fullfname = new_fname)
    elif option=='sum':
        data = sum_data(fname,verbose=verbose)
        new_fname = os.path.sep.join([dest_path,option,option+'_'+os.path.basename(fname)]) 
        save_h5(data, fullfname = new_fname)
    elif option=='mxs':
        data_max, data_sum = mxs_data(fname,verbose=verbose)
        max_fname = os.path.sep.join([dest_path,'max','max_'+os.path.basename(fname)])
        sum_fname = os.path.sep.join([dest_path,'sum','sum_'+os.path.basename(fname)]) 
        save_h5(data_max, fullfname = max_fname)
        save_h5(data_sum, fullfname = sum_fname)
    else:
        print('invalid option {}'.format(option))
        print(USAGE)
        
    


def main(args):


    verbose = True
    option = args.pop(0)
    rel_dest_path = '/../../../PROCESS/aj_log/'
    
    if option not in ['max', 'sum', 'mxs']:
        print('invalid option {}'.format(option))
        print(USAGE)
        sys.exit(0)
    todo_list = [[option,os.path.realpath(x),rel_dest_path,verbose] for x in args if x.find('.h5')>0]
    NOPROCESSES = min(8,len(todo_list))

    # operation_worker(todo_list[0])
    pool = Pool(NOPROCESSES, worker_init(os.getpid()))
    pool.map(operation_worker,todo_list)
    pool.close()
    pool.join()


if __name__ == '__main__':
    

    args=sys.argv[1:]
    
    main(args)
