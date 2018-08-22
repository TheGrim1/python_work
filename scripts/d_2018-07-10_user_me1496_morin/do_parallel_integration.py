
from multiprocessing import Pool
import os, sys
import integrator_1d as integrator
import subprocess
import h5py
import numpy as np
import time

sys.path.append('/data/id13/inhouse2/AJ/skript')
import fileIO.hdf5.h5_tools as ht
from pythonmisc.worker_suicide import worker_init

def merge_integrated_data(source_fname_list, dest_path, mapshape, no_frames):

    dest_fname = os.path.realpath(dest_path +'/'+os.path.basename(ht.parse_master_fname(source_fname_list[0])))

    if no_frames!=mapshape[0]*mapshape[1]:
        print('illegal number of frames {} for mapshape:'.format(no_frames))
        print(mapshape)
        sys.exit(0)
    # for i in range(21):
    #     for j in range(620):
    #         indexes.append([i,j])
    
    with h5py.File(dest_fname, 'w') as dest_h5:
        with h5py.File(source_fname_list[0],'r') as first_h5:
      
            entry_group = dest_h5.create_group('entry')
            entry_group.attrs['NX_class'] = 'NXentry'
            dest_group = dest_h5.create_group('entry/integrated')
            dest_group.attrs['NX_class'] = 'NXcollection'
            dest_grouppath = 'entry/integrated'
            
            qrad_group = dest_h5[dest_grouppath].create_group(name='q_radial')
            qrad_group.attrs['NX_class'] = 'NXdata'
            qradvert_group = dest_h5[dest_grouppath].create_group(name='q_radial_vert')
            qradvert_group.attrs['NX_class'] = 'NXdata'
            
            first_h5.copy(dest_grouppath+'/axes',dest_h5[dest_grouppath])
            axes_group = dest_h5[dest_grouppath]['axes']
            ds_y = axes_group.create_dataset('y',data=np.linspace(0,619,620))
            ds_y.attrs['units'] ='um'
            ds_z = axes_group.create_dataset('z',data=np.linspace(0,20*5,21))
            ds_z.attrs['units'] ='um'
            datashape = (no_frames, axes_group['q_radial'].shape[0])

            I_shape = (mapshape[0],mapshape[1],datashape[1])
            I_ds = dest_group['q_radial_vert'].create_dataset('I',shape=I_shape,dtype=np.float32)
            dest_group['q_radial_vert/q_radial'] = dest_group['axes/q_radial']
            dest_group['q_radial_vert/z'] = dest_group['axes/z']
            dest_group['q_radial_vert/y'] = dest_group['axes/y']
            dest_group['q_radial_vert'].attrs['signal'] = 'I'
            dest_group['q_radial_vert'].attrs['axes'] = ['z','y','q_radial']
            # dest_group['q_radial_vert/q_radial'].attrs['units'] = 'q_nm^-1'

            I_vert_ds = dest_group['q_radial'].create_dataset('I',shape=I_shape,dtype=np.float32)
            dest_group['q_radial/q_radial'] = dest_group['axes/q_radial']
            dest_group['q_radial/z'] = dest_group['axes/z']
            dest_group['q_radial/y'] = dest_group['axes/y']
            dest_group['q_radial'].attrs['signal'] = 'I'
            dest_group['q_radial'].attrs['axes'] = ['z','y','q_radial']
            # dest_group['q_radial/q_radial'].attrs['units'] = 'q_nm^-1'

        I_data = np.zeros(shape=(datashape[0],datashape[1]),dtype=np.float32)
        I_vert_data = np.zeros(shape=(datashape[0],datashape[1]),dtype=np.float32)

        done_frames = 0
        for i, source_fname in enumerate(source_fname_list):
            with h5py.File(source_fname,'r') as source_h5:
                source_data = np.asarray(source_h5[dest_grouppath]['q_radial/I'])
                I_data[done_frames:done_frames+source_data.shape[0]] = source_data
                
                source_vert_data = np.asarray(source_h5[dest_grouppath]['q_radial_vert/I'])
                I_vert_data[done_frames:done_frames+source_data.shape[0]] = source_vert_data
                done_frames += source_data.shape[0]

        
        I_ds[:] = I_data.reshape(I_shape)
        I_vert_ds[:] = I_vert_data.reshape(I_shape)

    print('\n\nDONE')
    print('merged data in \n')
    print(dest_fname)
    

def start_from_masterfile(master_fname):
    master_starttime = time.time()
    noprocesses = 20
    tpl = ht.parse_data_fname_tpl(master_fname)
    
    data_fname_list = [os.path.realpath(tpl.format(x)) for x in range(10)]
    print(data_fname_list)
    data_fname_list = [os.path.realpath(x) for x in data_fname_list if os.path.exists(x)]

    mapshape = (21,620)
    with h5py.File(master_fname,'r') as master_h5:
        datashape = ht.get_datagroup_shape(master_h5['entry/data/'])[0]
    no_frames = datashape[0]
    if no_frames!=mapshape[0]*mapshape[1]:
        print('illegal number of frames {} for mapshape:'.format(no_frames))
        print(mapshape)

    print(no_frames,mapshape)
    print(mapshape)
    
    dest_path ='/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-10_user_me1496_morin/PROCESS/SESSION_INTEGRATE/integrated/'
    tmp_dest_path = dest_path+ 'tmp/'
    
    poni_fname = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-10_user_me1496_morin/PROCESS/SESSION27/calib_detx_795/calib_detx_795.poni'

    ## masked values =1
    mask_fname = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-10_user_me1496_morin/PROCESS/SESSION27/mask.edf'
    mask_vert_fname = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-10_user_me1496_morin/PROCESS/SESSION27/mask_vert.edf'

    print('integrating datafiles:')
    for x in data_fname_list:
        print(x)

    print('poni_fname:\n{}'.format(poni_fname))
    print('mask_fname:\n{}'.format(mask_fname))
    
    # data_fname_list = ['/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-21_user_ma4110_dufour//DATA/AUTO-TRANSFER/eiger1/cell_cycleb3_65_1070_data_000001.h5', '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-21_user_ma4110_dufour/DATA/AUTO-TRANSFER/eiger1/cell_cycled0_31_1156_data_000001.h5','/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-21_user_ma4110_dufour/DATA/AUTO-TRANSFER/eiger1/cell_cycled0_117_1242_data_000001.
    
    dest_fname_list = [tmp_dest_path + 'integrated_' + os.path.basename(fname) for fname in data_fname_list]
    dest_fname_list.sort()
    
    todo_list = []
    if not os.path.exists(tmp_dest_path):
        os.mkdir(tmp_dest_path)
    print('starting {} processes to integrate'.format(noprocesses))
    
    for data_fname, dest_fname in zip(data_fname_list,dest_fname_list):
        todo_list.append([data_fname,
                          dest_fname,
                          poni_fname,
                          mask_fname,
                          mask_vert_fname,
                          True])
        print(data_fname)

    # # DEBUG:
    # integrator.do_integration(todo_list[0])
        
    pool = Pool(max(noprocesses,len(todo_list)),worker_init(os.getpid()))
    pool.map_async(integrator.do_integration,todo_list)
    pool.close()
    pool.join()

    merge_integrated_data(dest_fname_list, dest_path, mapshape=mapshape, no_frames=no_frames)
    master_endtime = time.time()
    master_time = (master_endtime - master_starttime)
    print('='*25)
    print('\nTotal time taken for integration of {} frames = {}'.format(no_frames, master_time))
    print(' = {} Hz\n'.format(no_frames/master_time))
    print('='*25) 

def main(args):

    data_fname_list = [os.path.realpath(x) for x in args if x.find('_data_')>0]
    path = os.path.dirname(data_fname_list[0])+'/'
    master_fname_list = [path+ht.parse_master_fname(os.path.basename(x)) for x in data_fname_list]
    for i, master_fname in enumerate(master_fname_list):
        print('**'*50+'\nstaring on masterfile {} out of {}:'.format(i+1,len(master_fname_list)))
        print(master_fname+'\n')
        start_from_masterfile(master_fname)
        
    
if __name__ == '__main__':
    
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc.  \n2)
python <thisfile.py> -f <file containing args as lines> \n3) find
<*yoursearch* -> arg1 etc.> | python <thisfile.py> """


    
    args = []
    if len(sys.argv) > 1:
        if sys.argv[1].find("-f")!= -1:
            f = open(sys.argv[2])
            for line in f:
                args.append(line.rstrip())
        else:
            args=sys.argv[1:]
    else:
        f = sys.stdin
        for line in f:
            args.append(line.rstrip())
    
    main(args)
