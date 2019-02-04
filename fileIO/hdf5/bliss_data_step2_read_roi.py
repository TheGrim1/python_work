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
import fileIO.hdf5.workers.bliss_align_data_worker as badw
import pythonmisc.pickle_utils as pu   


def do_align_diffraction_data(dest_fname, source_fname, troi_dict, mask_fname, no_processes=4, verbose=False):
    parallel_align_diffraction(dest_fname, source_fname, troi_dict, mask_fanme, no_processes, verbose)
    collect_align_diffraction(dest_fname, verbose)


def parallel_align_diffraction(dest_fname, source_fname, troi_dict, mask_fname, no_processes=4, verbose=False):
    '''
    memory restricted no_process see size of data to be aligned
    '''

    align_starttime = time.time()
    total_datalength = 0
    print('aligning diffraction data from\n{}'.format(dest_fname))
    todo_list = []
    curr_dir = os.path.dirname(dest_fname)
    alignmap_dir = curr_dir+os.path.sep+'diff_aligned'
    
    if os.path.exists(alignmap_dir):
        rmtree(alignmap_dir)
    
    os.mkdir(alignmap_dir)
        
    subdest_fname_tpl = alignmap_dir+os.path.sep+'{}'+os.path.sep+'single_map_{:08d}.h5'

    phi_h5path = 'axes/phi'
    
    with h5py.File(dest_fname) as dest_h5:

        diff_g = dest_h5['merged_data'].create_group('diffraction')
        shift = list(np.asarray(dest_h5['merged_data/alignment/shift']))

        troi_list=[]
        for troi_name, troi in troi_dict.items():
            troi_g = diff_g.create_group(troi_name)
            troi_g.create_dataset('troi',data=troi)
            troi_list.append([troi_name, troi, troi_g])

            
        with h5py.File(source_fname,'r') as source_h5:
            
            phi_list = phi_list = [[data_g[phi_h5path].value, key] for key, data_g in source_h5.items()]
            phi_list.sort()
            # Theta_list = [[float(key), value.value] for key, value in dest_h5['integrated_files'].items()]
            shift_dict = dict(zip([phi_pos for phi_pos,_ in  phi_list],np.asarray(shift)))
            print(shift_dict)

            axes_g = dest_h5['merged_data/axes']
            map_shape = (axes_g['y'].shape[0],axes_g['x'].shape[0])

            for troi_name, troi, troi_g in troi_list:
                single_g = troi_g.create_group('single_scans')                
                troi_dir = alignmap_dir+os.path.sep+troi_name

                if not os.path.exists(troi_dir):
                    os.mkdir(troi_dir)
                    
                for phi_pos, source_grouppath in phi_list:                 
                    # new dest_fname needs to be made to circumvent parrallelism issues
                    subdest_fname = subdest_fname_tpl.format(troi_name,int(1000*phi_pos))

                    phi_g = single_g.create_group(os.path.basename(subdest_fname))
                    phi_g.create_dataset('phi',data=phi_pos)
                    phi_g.create_dataset('path',data=subdest_fname)

                    total_datalength += map_shape[0]*map_shape[1]

                    todo_list.append([subdest_fname,
                                      source_fname,
                                      source_grouppath,
                                      troi,
                                      mask_fname,
                                      phi_pos,
                                      map_shape,
                                      shift_dict[phi_pos],
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
            badw.bliss_align_data_worker(instruction)
        ## non parrallel version for one dataset and timing:
        #fdw.fit_data_worker(instruction_list[0])
    else:
        pool = Pool(no_processes, worker_init(os.getpid()))
        pool.map_async(badw.bliss_align_data_employer,instruction_list)
        pool.close()
        pool.join()

            
    align_endtime = time.time()
    align_time = (align_endtime - align_starttime)
    print('='*25)
    print('\ntime taken for aligning of {} frames = {}'.format(total_datalength, align_time))
    print(' = {} Hz\n'.format(total_datalength/align_time))
    print('='*25) 

def collect_align_diffraction_data(dest_fname,verbose):
    
    collect_starttime = time.time()
    print('collecting all the parallely processed data in {}'.format(dest_fname))
    curr_dir = os.path.dirname(dest_fname)
    alignmap_dir = curr_dir+os.path.sep+'diff_aligned/'

    with h5py.File(dest_fname) as dest_h5:
        diff_g = dest_h5['merged_data/diffraction']
        troi_list = [[key, value] for key, value in diff_g.items()]
        axes_g = dest_h5['merged_data/axes']

        for troi_name, dest_troi_g in troi_list:

            first = True
            for source_bname, phi_g in dest_troi_g['single_scans'].items():
                source_fname = phi_g['path'].value
                phi_pos = float(phi_g['phi'].value)
                
            
                if verbose:
                    print('reading {}'.format(source_bname))
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
            dest_troi_g.create_dataset('sum', data=curr_sum)
            dest_troi_g.create_dataset('max', data=curr_max)
                        
        dest_h5.flush()

            
    collect_endtime = time.time()
    collect_time = (collect_endtime - collect_starttime)
    print('='*25)
    print('\ntime taken for collecting all frames = {}'.format(collect_time))
    print('='*25) 
        
    
    
def main(preview_fname, saving_name, dest_path, mask_fname, troi_dict):
    verbose = True

    dest_fname = os.path.realpath(dest_path + saving_name + '_merged.h5')
    
    parallel_align_diffraction(dest_fname, source_fname=preview_fname, troi_dict=troi_dict, mask_fname=mask_fname, no_processes=10, verbose=False)
    
    collect_align_diffraction_data(dest_fname, verbose)
    

if __name__ == '__main__':
        
    # session_name = 'alignment'
    # saving_name = 'kmap_rocking'
    # map_shape = (140,80)
    
    # session_name = 'day_two'
    # saving_name = 'kmap_and_cen_4b'

    # troi_dict = {'red':np.asarray([[995,210],[1018-995,235-210]]),
    #              'blue':np.asarray([[497,1192],[513-497,1232-1192]]),
    #     	 'green':np.asarray([[760,1800],[800-760,1840-1800]])}

    session_name = 'day_two'
    saving_name = 'kmap_and_cen_3b'
    troi_dict = {'black':np.asarray([[1306,600],[1327-1306,636-600]]),
                 'yellow':np.asarray([[1505,1404],[1523-1505,1422-1404]]),
        	 'cyan':np.asarray([[392,1685],[409-392, 1702-1685]])}

    
    # session_name = 'alignment'
    # saving_name = 'kmap_rocking4'
    # troi_dict = {'red':np.asarray([[1997,645],[2133-1997,675-645]]),
    #              'blue':np.asarray([[1262,1780],[1284-1262,1800-1780]])}

    session_path = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/DATA/'+session_name+ '/eh3/'

    dest_path = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/previews/'+session_name +'/'
    mask_fname = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/jupyter_output/mask_neg.edf'
    preview_file = dest_path +'/'+ saving_name + '/' + saving_name + '_preview.h5'
    
    main(preview_file, saving_name, dest_path, mask_fname, troi_dict)
