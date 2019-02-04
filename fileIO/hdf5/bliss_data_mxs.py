import h5py
import sys, os
import numpy as np
import time
import shutil
import glob
from multiprocessing import Pool
import fabio

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
from pythonmisc.worker_suicide import worker_init
import fileIO.edf.edfIO as open_edf
from fileIO.hdf5.bliss_data_preview import find_kmaps_h5
from fileIO.hdf5.workers import bliss_mxs_worker as bmxs

def main(session_path, dest_path, saving_name, session, expected_len, mask_fname, verbose=True):

    starttime = time.time()
    total_datalength = 0

    ### get the need info:
    no_processes = 30

    tmp_folder = dest_path+'/{}_tmp/'.format(os.getpid())
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.mkdir(tmp_folder)
    
    raw_data_path = 'entry_0000/instrument/E-08-0106/image_data'
    kmaps_fname_dict = find_kmaps_h5(session_path,saving_name)
    todo_list = []
    for sname, sname_dict in kmaps_fname_dict.items():
        for scan_name, source_fname in sname_dict['kmaps'].items():
            todo = []
            with h5py.File(source_fname,'r') as source_h5:
                data = source_h5[raw_data_path]
                data_shape = data.shape
                if data_shape[0] == expected_len:
                    total_datalength += expected_len
                    
                    todo_list.append([source_fname,
                                      tmp_folder,
                                      verbose])
                    
                    print('appending {}'.format(source_fname))
                else:
                    print('rejecting {}'.format(source_fname))
                    

    instruction_list = []
    for i,todo in enumerate(todo_list):
        instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=False, counter=i)
        instruction_list.append(instruction_fname)       

    if no_processes==1:
        for i, instruction in enumerate(instruction_list):
            print('running in single process, loop {}'.format(i))
            qrw.bliss_mxs_worker(instruction)

    else:
        pool = Pool(min(no_processes,len(instruction_list)), worker_init(os.getpid()))
        pool.map_async(bmxs.bliss_mxs_employer,instruction_list)
        pool.close()
        pool.join()

    endreadtime = time.time()
    total_read_time = (endreadtime - starttime)
    print('='*25)
    print('\ntime taken for partial summing of {} datasets = {}'.format(total_datalength, total_read_time))
    print(' = {} Hz\n'.format(total_datalength/total_read_time))
    print('='*25) 
             
    dest_fname = dest_path + saving_name + '_mxs.h5'

    sum_fname_list = glob.glob(tmp_folder+'*sum.edf')
    max_fname_list = glob.glob(tmp_folder+'*max.edf')
    
    mask = fabio.open(mask_fname).data

    with h5py.File(dest_fname,'w') as dest_h5:

        data_sum = np.zeros(shape = data_shape[1:],dtype=np.uint64)
        data_max = np.zeros(shape = data_shape[1:],dtype=np.uint32)
        
        for sum_fname in sum_fname_list:
            print('reading {}'.format(sum_fname))
            data_sum += fabio.open(sum_fname).data

        for max_fname in max_fname_list:
            print('reading {}'.format(max_fname))
            frame = fabio.open(max_fname).data
            data_max = np.where(data_max>frame, data_max, frame)

        mxs_g = dest_h5.create_group('mxs')
        mxs_g.create_dataset('sum',data=data_sum*mask)
        mxs_g.create_dataset('max',data=data_max*mask)

    shutil.rmtree(tmp_folder)

    endsumtime = time.time()
    total_time = (endsumtime - starttime)
    print('='*25)
    print('\ntime taken for full summing of {} datasets = {}'.format(total_datalength, total_time))
    print(' = {} Hz\n'.format(total_datalength/total_time))
    print('wrote {}'.format(dest_fname))
    print('='*25) 
             
if __name__ == '__main__':
        
    session_name = 'alignment'
    saving_name = 'kmap_rocking4'
    expected_len = 140*80

    # session_name = 'alignment'
    # saving_name = 'kmap_rocking5'
    # expected_len = 140*80

    # session_name = 'day_two'
    # saving_name = 'kmap_and_cen_4b'
    # expected_len = 120*80

    # session_name = 'day_two'
    # saving_name = 'kmap_and_cen_3b'
    # expected_len = 150*80

    
    mask_fname =  '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/jupyter_output/mask_neg.edf'

    
    session_path = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/DATA/'+session_name+ '/eh3/'
    dest_path = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/previews/'+session_name +'/'

    main(session_path, dest_path, saving_name, session_name, expected_len, mask_fname, verbose=True)
