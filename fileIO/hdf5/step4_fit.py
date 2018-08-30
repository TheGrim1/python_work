import h5py
import numpy as np
import sys, os
import datetime
import time

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

from pythonmisc.worker_suicide import worker_init
import pythonmisc.pickle_utils as pu

import fileIO.hdf5.workers.fit_data_worker as fdw

    
def do_fit_raw_diffraction_data(merged_fname, no_processes=10, verbose=False):

    fit_starttime = time.time()
    total_datalength = 0
    no_peaks = 4
    print('fitting diffraction data from\n'.format(merged_fname))
    todo_list = []
    curr_dir = os.path.dirname(merged_fname)
    tthtroi_dict={}
    diffmap_dir = curr_dir+os.path.sep+'temp_diff_fits'

    if os.path.exists(diffmap_dir):
        rmtree(diffmap_dir)
    
    os.mkdir(diffmap_dir)
        
    subdest_fname_tpl = diffmap_dir+os.path.sep+'{}'+os.path.sep+'single_map_fit_{:08d}.h5'
          
    with h5py.File(dest_fname) as dest_h5:
        
        Theta_list = [[float(key), value.value] for key, value in dest_h5['entry/integrated_files'].items()]
        axes = dest_h5['entry/merged_data/axes']
        
        with h5py.File(Theta_list[0][1]) as first_h5:
            troi_list = first_h5['entry/integrated/'].keys()

            for troiname in troi_list:
                tth_troi_dict.update(troiname, first_h5['entry/integrated/{}/axes/tthtroi'.format(troiname)])
                
                troi_dir = diffmap_dir+os.path.sep+troiname
                if not os.path.exists(troi_dir):
                    os.mkdir(troi_dir)


                for Theta, fname in Theta_list:
                    # new dest_fname needs to be made to circumvent parrallelism issues
                    subsource_fname = fname
                    subdest_fname = subdest_fname_tpl.format(troiname,int(1000*Theta))
                    subdest_grouppath = 'data'
                    subsource_grouppath = 'entry/integrated/{}'.format(troiname)
                    
                    mapshape = (axes['y'].shape[0],axes['x'].shape[0])

                    total_datalength+=mapshape[0]*mapshape[1]

                    todo_list.append([subdest_fname,
                                      subdest_grouppath,
                                      subsource_fname,
                                      subsource_grouppath,
                                      Theta,
                                      mapshape,
                                      no_peaks,
                                      verbose])

        dest_h5.flush()        
                    
    print('setup parallel proccesses to write to {}'.format(diffmap_dir))
    
    instruction_list = []
    for i,todo in enumerate(todo_list):
        #DEBUG:
        print('todo #{:2d}'.format(i))
        print(todo)
        instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
        instruction_list.append(instruction_fname)

    if no_processes==1:
        # DEBUG (change to employer for max performance
        for instruction in instruction_list:
            fdw.fit_data_employer(instruction)
        ## non parrallel version for one dataset and timing:
        #fdw.fit_data_worker(instruction_list[0])
    else:
        pool = Pool(no_processes,worker_init(os.getpid()))
        pool.map_async(fdw.fit_data_employer,instruction_list)
        pool.close()
        pool.join()
        
    print('collecting all the parallely processed data in '.format(dest_fname))
    with h5py.File(dest_fname) as dest_h5:
        diff_merged = dest_h5['entry/merged_data'].create_group('fit_raw_diffraction')
        diff_merged.attrs['NX_class'] = 'NXcollection'
        axes = dest_h5['entry/merged_data/axes']
        axes.create_dataset(name='peak_number', data=np.arange(no_peaks))
        axes.create_dataset(name='peak_parameter', data=['a','mu','sig'])

        
        for troiname in troi_list:
            axes.create_dataset(name='tthtroi_{}'.format(troiname), data=tth_troi_dict(troi))
            troi_merged = diff_merged.create_group(troiname)
            single_maps = troi_merged.create_group('single_maps')
            single_maps.attrs['NX_class'] = 'NXprocess'
            source_dir = diffmap_dir+os.path.sep+troiname
            fname_list = [source_dir+os.path.sep+x for x in os.listdir(source_dir) if x.find('.h5')]
            fname_list.sort()
            
            for i, subsource_fname in enumerate(fname_list):
                with h5py.File(subsource_fname,'r') as source_h5:
                    Theta = source_h5['data/Theta'].value
                    Theta_group = single_maps.create_group(name=str(Theta))
                    Theta_group.attrs['NX_class'] = 'NXdata'
                    Theta_group.attrs['signal'] = 'sum'
                    Theta_group.attrs['source_filename'] = subsource_fname

                    Theta_group.attrs['axes'] = ['x','y']
                    Theta_group['x'] = axes['x']
                    Theta_group['y'] = axes['y']
                    Theta_group['peak_number'] = axes['peak_number']
                    Theta_group['peak_parameter'] = axes['peak_parameter']
                    Theta_group['Theta'] = axes['Theta']
                    
                    Theta_group.create_dataset('sum', data=np.asarray(source_h5['data/sum']))
                    Theta_group.create_dataset('max', data=np.asarray(source_h5['data/max']))
                    Theta_group.create_dataset('chi_com', data=np.asarray(source_h5['data/chi_com']))
                    Theta_group.create_dataset('tth_com', data=np.asarray(source_h5['data/tth_com']))
                    Theta_group.create_dataset('chi_fit', data=np.asarray(source_h5['data/chi_fit']))
                    Theta_group.create_dataset('tth_fit', data=np.asarray(source_h5['data/tth_fit']))
                    Theta_group.create_dataset('2d_fit', data=np.asarray(source_h5['data/2d_fit']))
                    
        dest_h5.flush()
    
    fit_endtime = time.time()
    fit_time = (fit_endtime - fit_starttime)
    print('='*25)
    print('\ntime taken for fitting of {} frames = {}'.format(total_datalength, fit_time))
    print('='*25) 
    

   
def main():


    masterfolder = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_gpu2/'

if __name__ == "__main__":
    main()
