import time
import sys,os
import h5py
import numpy as np
import pyFAI
import fabio
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import integrate_data_worker as idw
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy

def do_integration(args):

    source_fname = args[0]
    dest_fname = args[1]
    poni_fname = args[2]
    mask_fname = args[3]
    verbose = args[4]    
    integ_starttime = time.time()
    total_datalength = 0
    noprocesses = 1

    source_datasetpath = 'entry/data/data'
    
    print('\nINTEGRATING, verbose = %s\n' % verbose)
    print('data file {}'.format(source_fname))

    with h5py.File(source_fname,'r') as source_h5:
        todolist = []
        source_dataset = source_h5[source_datasetpath]
        with h5py.File(dest_fname,'w') as dest_h5:

            entry_group = dest_h5.create_group('entry')
            entry_group.attrs['NXclass'] = 'NXentry'
            dest_group = dest_h5.create_group('entry/integrated')
            dest_group.attrs['NXclass'] = 'NXcollection'
            dest_grouppath = 'entry/integrated'

            datashape = source_dataset.shape
            # dtype = h5_file[raw_datasetpath].dtype
            dtype = np.float32
            datalength = datashape[0]

            npt_rad  = int(datashape[2])
            npt_azim = int(datashape[1])

            # setup groups and datasets
            qrad_group = dest_h5[dest_grouppath].create_group(name='q_radial')
            qrad_group.attrs['NXclass'] = 'NXdata' 
            chi_group = dest_h5[dest_grouppath].create_group(name='chi_azimuthal')
            chi_group.attrs['NXclass'] = 'NXdata'
            q2D_group = dest_h5[dest_grouppath].create_group(name='q_2D')
            q2D_group.attrs['NXclass'] = 'NXdata' 
            axes_group = dest_h5[dest_grouppath].create_group(name='axes')
            axes_group.attrs['NXclass'] = 'NXdata' 
        
            dest_group['q_radial'].create_dataset(name = 'I', dtype = dtype, shape=(datashape[0],npt_rad), compression='lzf', shuffle=True)
            dest_group['chi_azimuthal'].create_dataset(name = 'I', dtype = dtype, shape=(datashape[0],npt_azim), compression='lzf', shuffle=True)
            dest_group['q_2D'].create_dataset(name = 'data', dtype = dtype, shape=(datashape[0],npt_azim,npt_rad), compression='lzf', shuffle=True)



            dest_h5.flush()

            if mask_fname != None:
                mask = fabio.open(mask_fname).data
            else:
                mask = np.zeros(shape = (datashape[1],datashape[2]))

            dest_group.create_dataset(name = 'mask', dtype = np.bool, data=mask, compression='lzf', shuffle=True)
            
                
            # do one integration to get the attributes out of ai:
            ai = pyFAI.AzimuthalIntegrator()              
            ai.load(poni_fname)
            ai.setChiDiscAtZero()

            frame = 0
            raw_data = source_dataset[frame]

            q_data, qrange, qazimrange  = ai.integrate2d(data = raw_data, mask=mask, npt_azim = npt_azim ,npt_rad = npt_rad , unit='q_nm^-1')

            if verbose:
                print('q-unit = {}, qtroi = '.format('q_nm^-1'))
                print('qrange    = from %s to %s'% (max(qrange),min(qrange)))
                print('azimrange = from %s to %s'% (max(qazimrange),min(qazimrange)))

            dest_group['axes'].create_dataset(name = 'frame_no', data=np.arange(datashape[0]))
            dest_group['axes'].create_dataset(name = 'q_radial', data=qrange)
            dest_group['axes/q_radial'].attrs['units'] = 'q_nm^-1'
            dest_group['axes'].create_dataset(name = 'chi_azim', data=qazimrange)
            dest_group['axes/chi_azim'].attrs['units'] = 'deg'


            dest_group['q_radial/frame_no'] = dest_group['axes/frame_no']
            dest_group['q_radial/q_radial'] = dest_group['axes/q_radial']
            dest_group['q_radial'].attrs['signal'] = 'I'
            dest_group['q_radial'].attrs['axes'] = ['frame_no','q_radial']
            dest_group['q_radial/q_radial'].attrs['units'] = 'q_nm^-1'

            dest_group['chi_azimuthal/frame_no'] = dest_group['axes/frame_no']
            dest_group['chi_azimuthal/azim_range'] = dest_group['axes/chi_azim']
            dest_group['chi_azimuthal'].attrs['signal'] = 'I'
            dest_group['chi_azimuthal'].attrs['axes'] = ['frame_no','azim_range']

            dest_group['q_2D/frame_no'] = dest_group['axes/frame_no']
            dest_group['q_2D/q_radial'] = dest_group['axes/q_radial']
            dest_group['q_2D/azim_range'] = dest_group['axes/chi_azim']
            dest_group['q_2D'].attrs['signal'] = 'data'
            dest_group['q_2D'].attrs['axes'] = ['frame_no', 'azim_range', 'q_radial']

            # now setup for integration of all frames in parallel:

            datalength = datashape[0]

            total_datalength += datalength

            # split up the indexes to be processed per worker:
            index_interval = int(datalength / noprocesses)
            source_index_list = [range(i*index_interval,(i+1)*index_interval) for i in range(noprocesses)]
            source_index_list[noprocesses-1] = range((noprocesses-1)*index_interval,datalength)
            target_index_list = source_index_list            


            [todolist.append([dest_fname,
                              dest_grouppath,
                              target_index_list[i],
                              source_fname,
                              source_datasetpath,
                              source_index_list[i],
                              poni_fname,
                              [npt_azim, npt_rad],
                              verbose]) for i in range(len(target_index_list))]

            dest_h5.flush()    
    instruction_list = []

    for i,todo in enumerate(todolist):
        instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
        instruction_list.append(instruction_fname)

    if noprocesses==1:
        # DEBUG (change to employer for max performance
        for instruction in instruction_list:
            idw.integrate_data_employer(instruction)
        ## non parrallel version for one dataset and timing:
        #idw.integrate_data_worker(instruction_list[0])
    else:
        pool = Pool(processes=noprocesses)
        pool.map_async(idw.integrate_data_employer,instruction_list)
        pool.close()
        pool.join()


    integ_endtime = time.time()
    integ_time = (integ_endtime - integ_starttime)
    print('='*25)
    print('\ntime taken for integration of {} frames = {}'.format(total_datalength, integ_time))
    print(' = {} Hz\n'.format(total_datalength/integ_time))
    print('='*25) 

