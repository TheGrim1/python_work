from __future__ import print_function
from __future__ import absolute_import


USAGE = """ \n1) python <thisfile.py> <OPTIONS ><arg1> <arg2> etc. 
\n2) find <*yoursearch* -> arg1 etc.> | python <OPTIONS> <thisfile.py> 
\n--------
\n operates on each h5 (datafile), ouputs a file each
\n<OPTIONS>:
\n    mxs   - max and sum
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
import glob
from shutil import rmtree
import fabio
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
# local

# from plot_h5 import plot_h5
from pythonmisc.worker_suicide import worker_init
from fileIO.hdf5.workers import mxs_worker as mxsw
import pythonmisc.pickle_utils as pu
import fileIO.spec.spec_tools as st
from fileIO.edf.save_edf import save_edf
from fileIO.hdf5.h5_tools import do_merge_path


def operation_worker(fname, dest_path, no_processes=9, verbose=False):
    if not os.path.exists(fname):
        print('path not found')
        print('path fname {}'.format(fname))
        sys.exit(0)
    if os.path.exists(dest_path):
        if verbose:
            print('removing {}'.format(dest_path))
        rmtree(dest_path)
    os.mkdir(dest_path)
    
    max_path = dest_path+'/single_max/'
    sum_path = dest_path+'/single_sum/'    
    os.mkdir(max_path)
    os.mkdir(sum_path)
    all_max_path = dest_path+'/all_max/'
    all_sum_path = dest_path+'/all_sum/'    
    os.mkdir(all_max_path)
    os.mkdir(all_sum_path)

    
    
    if verbose:
        print('pid: {} running {}'.format(os.getpid(), fname))

    with h5py.File(fname,'r') as source_h5:
        eta_list = []
        data_list = []
        for key,scan in source_h5.items():
            scan_header = scan['instrument/specfile/scan_header'].value
            print(scan_header)
            eta_list.append(st.get_ID01_rotations_from_scan_header(scan_header)['eta'])
            data_list.append(key)

        # sort by eta value
        sort_list = zip(eta_list,data_list)
        sort_list.sort()
        eta_list = [x for x,y in sort_list]
        data_list = [y for x,y in sort_list]

    all_sum_list = []
    all_max_list = []
    todo_list = []
    for path in data_list:
        data_path = path +'/instrument/detector/data'

        sum_dest = sum_path + path +'_sum.edf'
        max_dest = max_path + path +'_max.edf'
        all_sum_list.append(sum_dest)
        all_max_list.append(max_dest)
        todo_list.append([fname,
                          data_path,
                          0,
                          60,
                          sum_dest,
                          max_dest,
                          verbose])

    instruction_list=[]
    for i,todo in enumerate(todo_list):
        #DEBUG:
        print('todo #{:2d}:\n  -> {}'.format(i,todo[0]))
        instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
        instruction_list.append(instruction_fname)

    if no_processes==1:
        for instruction in instruction_list:
            mxsw.align_data_worker(instruction)
        ## non parrallel version for one dataset and timing:
        #fdw.fit_data_worker(instruction_list[0])
    else:
        pool = Pool(no_processes, worker_init(os.getpid()))
        pool.map_async(mxsw.mxs_employer,instruction_list)
        pool.close()
        pool.join()

    all_sum = np.zeros_like(fabio.open(all_sum_list[0]).data)
    for fname in all_sum_list:
        all_sum+= fabio.open(fname).data
    save_edf(all_sum, all_sum_path + path + '_all.edf')
        
    all_max = np.zeros_like(fabio.open(all_max_list[0]).data)
    for fname in all_max_list:
        all_max+= fabio.open(fname).data
    save_edf(all_max, all_max_path + path + '_all.edf')
                    
        
def main(args):
    
    verbose = True
    fname = args[0]
    rel_dest_path = '/' + os.path.basename(fname).split('.')[0] + '/mxs/'
    
    dest_path = os.path.realpath('../'+(os.path.dirname(fname)+rel_dest_path))
    print(dest_path)
    no_processes = 9
    operation_worker(fname, dest_path, no_processes,verbose)
    do_merge_path(dest_path,verbose)

    
if __name__ == '__main__':    

    args=sys.argv[1:]
    
    main(args)
