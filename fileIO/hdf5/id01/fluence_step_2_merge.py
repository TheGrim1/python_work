import h5py
import sys, os
import numpy as np
import time
import glob
from multiprocessing import Pool
import datetime
from shutil import rmtree

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

from pythonmisc.worker_suicide import worker_init
import fileIO.hdf5.workers.id01_align_worker as adw
import pythonmisc.pickle_utils as pu   


def do_align_diffraction_data(dest_fname, source_fname, troi_dict, mask_fname, no_processes=4, verbose=False):
    parallel_align_diffraction(dest_fname, source_fname, troi_dict, mask_fname, no_processes, verbose)
    collect_align_diffraction(dest_fname, verbose)


def parallel_align_diffraction(working_dir, dest_fname, no_processes=4, verbose=False):
    '''
    memory restricted no_process see size of data to be aligned
    '''
    
    align_starttime = time.time()
    total_datalength = 0
    print('aligning diffraction data in {}'.format(working_dir))
    todo_list = []
    
    alignmap_dir = working_dir + '/single_scans/'
    
    if os.path.exists(alignmap_dir):
        rmtree(alignmap_dir)
    
    os.mkdir(alignmap_dir)
        

    subdest_fname_tpl = alignmap_dir + '{}.h5'
    source_fname = glob.glob(working_dir + '/aligned/*_aligned.h5')[0]
    
    with h5py.File(dest_fname) as dest_h5:

        dest_merged_g = dest_h5.create_group('merged_data')
        dest_diff_g = dest_merged_g.create_group('diffraction')
        single_g = dest_diff_g.create_group('single_scans')
        
        with h5py.File(source_fname,'r') as source_h5:

            shift = list(np.asarray(source_h5['merged_data/alignment/shift']))
            merged_g = source_h5['merged_data']
            axes_g = merged_g['axes']

            eta_list = list(axes_g['eta'])
            phi_list = list(axes_g['phi'])
            del_list = list(axes_g['del'])
            angles = zip(eta_list, phi_list, del_list)
            
            scan_source_fnames = glob.glob(working_dir + '/mpx/*.edf.gz')
            scan_source_fnames.sort()            

            source_h5.copy('merged_data/axes',dest_merged_g,'axes')
            source_h5.copy('merged_data/alignment',dest_merged_g,'alignment')
            alignment_counter = source_h5['merged_data/alignment/alignment_parameters'].attrs['signal']
            source_h5.copy('merged_data/{}'.format(alignment_counter),dest_merged_g,alignment_counter)
            
            # axes_g = source_h5['merged_data/axes']
            # for key, val in axes_g.items():
            #     dest_axes_g.create_dataset(key,data=val)
            
            map_shape = (axes_g['y'].shape[0],axes_g['x'].shape[0])


            for i, scan_fname  in enumerate(scan_source_fnames):                 
                # new dest_fname needs to be made to circumvent parrallelism issues
                single_name = 'scan_{:06d}'.format(i) 
                subdest_fname = subdest_fname_tpl.format(single_name)
                phi_g = single_g.create_group(single_name)
                phi_g.create_dataset('angles', data=np.asarray(angles[i]))
                phi_g.create_dataset('path', data=subdest_fname)

                total_datalength += map_shape[0]*map_shape[1]

                todo_list.append([subdest_fname,
                                  scan_fname,
                                  angles[i],
                                  map_shape,
                                  shift[i],
                                  verbose])       

    print('setup parallel proccesses to write to {}'.format(alignmap_dir))

    instruction_list = []
    for i,todo in enumerate(todo_list):
        #DEBUG:
        print('todo #{:2d}:\n  -> {}'.format(i,todo[0]))
        instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
        instruction_list.append(instruction_fname)

    if no_processes==1:
        for instruction in instruction_list:
            adw.id01_align_data_worker(instruction)
        ## non parrallel version for one dataset and timing:
        #fdw.fit_data_worker(instruction_list[0])
    else:
        pool = Pool(no_processes, worker_init(os.getpid()))
        pool.map_async(adw.id01_align_data_employer,instruction_list)
        pool.close()
        pool.join()

            
    align_endtime = time.time()
    align_time = (align_endtime - align_starttime)
    print('='*25)
    print('\ntime taken for aligning of {} frames = {}'.format(total_datalength, align_time))
    print(' = {} Hz\n'.format(total_datalength/align_time))
    print('='*25)

    return alignmap_dir

def collect_align_diffraction_data(dest_fname, alignmap_dir, verbose):
    
    collect_starttime = time.time()
    print('collecting all the parallely processed data in {}'.format(dest_fname))
    curr_dir = os.path.dirname(dest_fname)

    with h5py.File(dest_fname) as dest_h5:
        diff_g = dest_h5['merged_data/diffraction']
        axes_g = dest_h5['merged_data/axes']


        first = True
        for phi_g in diff_g['single_scans'].values():
            source_fname = phi_g['path'].value
            angles = np.asarray(phi_g['angles'])


            if verbose:
                print('reading {}'.format(source_fname))
            with h5py.File(source_fname, 'r') as source_h5:

                source_data = source_h5['shifted_data']

                data = np.asarray(source_data['data'])
                data_sum = np.asarray(source_data['sum'])
                data_max = np.asarray(source_data['max'])

                phi_g.create_dataset('data',data=data)
                phi_g.create_dataset('sum',data=data_sum)
                phi_g.create_dataset('max',data=data_max)
                phi_g['x'] = axes_g['x']
                phi_g['y'] = axes_g['y']
                phi_g.attrs['NX_class'] = 'NXdata'
                phi_g.attrs['signal'] = 'sum'

                if first:
                    first=False
                    curr_max = np.zeros_like(data_max)
                    curr_sum = np.zeros_like(data_sum)

                curr_max = np.where(curr_max<data_max, data_max, curr_max)
                curr_sum += data_sum
                
        diff_g.create_dataset('sum', data=curr_sum)
        diff_g.create_dataset('max', data=curr_max)
                        
        dest_h5.flush()

            
    collect_endtime = time.time()
    collect_time = (collect_endtime - collect_starttime)
    print('='*25)
    print('\ntime taken for collecting all frames = {}'.format(collect_time))
    print('='*25) 
        
    
    
def main(working_dir):
    verbose = True

    dest_path = working_dir + '/merged/'
    dest_fname = dest_path + working_dir.split(os.path.sep)[-1] + '_merged.h5'

    if os.path.exists(dest_path):
        rmtree(dest_path)
    os.mkdir(dest_path)

    
    alignmap_dir = parallel_align_diffraction(working_dir, dest_fname, no_processes=1, verbose=False)
    
    collect_align_diffraction_data(dest_fname, alignmap_dir, verbose)
    

if __name__ == '__main__':
        
    working_dir_list = glob.glob('/data/id13/inhouse2/AJ/data/ma3576/id01/analysis/fluence/KMAPS/*')
    working_dir_list = [x for x in working_dir_list if os.path.isdir(x)]
    # = '/data/id13/inhouse2/AJ/data/ma3576/id01/analysis/fluence/KMAPS/KMAP_2018_02_12_191535'

    
    for working_dir in working_dir_list:
        main(working_dir)
