import h5py
import numpy as np
import sys, os
import datetime
import time
from multiprocessing import Pool
from scipy.ndimage import shift as ndshift
from shutil import rmtree
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

import simplecalc.image_align_elastix as ia
import pythonmisc.pickle_utils as pu
import fileIO.hdf5.workers.align_data_worker as adw
from fileIO.datafiles.open_data import open_data
from pythonmisc.worker_suicide import worker_init

def find_my_h5_files(masterfolder):
    
    folder_list = [masterfolder+os.path.sep+x for x in os.listdir(masterfolder)]
    folder_list=  [x for x in folder_list if os.path.isdir(x)]
    
    my_h5_fname_list=[]
    
    for folder in folder_list:
        
        for fname in os.listdir(folder):
            if fname.find('integrated.h5')>0:
                my_h5_fname_list.append(folder+os.path.sep+fname)

    my_h5_fname_list.sort()
    return my_h5_fname_list


def do_fluo_merge(dest_fname, fluo_counter = 'ball01', verbose=False):

    print('reading fluorescence from counter {} data into file'.format(fluo_counter))
    print(dest_fname)
    
    with h5py.File(dest_fname) as dest_h5:
        
        Theta_list = [[float(key), value.value] for key, value in dest_h5['entry/integrated_files'].items()]
        Theta_list.sort()
        fname_list = [x[1] for x in Theta_list]
        merged_data = dest_h5['entry/merged_data']
        fluo_merged = dest_h5['entry/merged_data/fluorescence']


        # setup groups in dest_h5
        with h5py.File(fname_list[0]) as source_h5:

            speccmd_list = source_h5['entry/instrument/spec/title'].value.split(' ')
            speccmd_list = [x for x in speccmd_list if x != '']
            print('found spec command {}'.format(' '.join(speccmd_list)))
            x_mot = speccmd_list[1]
            y_mot = speccmd_list[5]

            x_pts = int(speccmd_list[4])+1
            y_pts = int(speccmd_list[8])+1
            mapshape= (y_pts, x_pts)
            Theta_pts = len(fname_list)

            # print('xmot {}'.format(x_mot))
            # print('ymot {}'.format(y_mot))
            
            x = np.asarray(source_h5['entry/instrument/measurement/{}'.format(x_mot)].value).reshape(mapshape)[0,:]
            y = np.asarray(source_h5['entry/instrument/measurement/{}'.format(y_mot)].value).reshape(mapshape)[:,0]

            axes = dest_h5['entry/merged_data'].create_group('axes')
            axes.attrs['NX_class'] = 'NXcollection'
            axes.create_dataset('Theta', dtype= y.dtype, shape = (Theta_pts,))
            axes.create_dataset('x' ,data=x)
            axes.create_dataset('y', data=y)
        
            fluo_ori = fluo_merged.create_group('fluo_original')
            fluo_ori.attrs['NX_class'] = 'NXdata'
            fluo_ori.attrs['signal'] = 'XRF_norm'
            fluo_ori.attrs['axes'] = ['Theta','y','x']
            fluo_ori['Theta'] = axes['Theta']
            fluo_ori['x'] = axes['x']
            fluo_ori['y'] = axes['y']
            fluo_ori.create_dataset(name='XRF_norm', dtype=y.dtype, shape=(Theta_pts, y_pts, x_pts), compression='lzf', shuffle=False)

            fluo_aligned = fluo_merged.create_group('fluo_aligned')
            fluo_aligned.attrs['NX_class'] = 'NXdata'
            fluo_aligned.attrs['signal'] = 'XRF_norm'
            fluo_aligned.attrs['axes'] = ['Theta','y','x']
            fluo_aligned['Theta'] = axes['Theta']
            fluo_aligned['x'] = axes['x']
            fluo_aligned['y'] = axes['y']

            single_maps = fluo_merged.create_group('single_maps')
            single_maps.attrs['NX_class'] = 'NXprocess'
    
        for i,[Theta, fname] in enumerate(Theta_list):
            with h5py.File(fname) as source_h5:

                print('reading no {} of {}'.format(i+1,Theta_pts))
                print('Theta {}, file {}'.format(Theta,fname))
                
                fluo_data = np.asarray(source_h5['entry/instrument/measurement/{}'.format(fluo_counter)].value).reshape(mapshape)
                mon_data = np.asarray(source_h5['entry/instrument/measurement/{}'.format('Monitor')].value).reshape(mapshape)

                Theta_group = single_maps.create_group(name=str(Theta))
                Theta_group.attrs['NX_class'] = 'NXdata'
                Theta_group.attrs['signal'] = 'XRF'
                Theta_group.attrs['source_filename'] = fname
                Theta_group.attrs['axes'] = ['x','y']
                
                x = np.asarray(source_h5['entry/instrument/measurement/{}'.format(x_mot)].value).reshape(mapshape)
                y = np.asarray(source_h5['entry/instrument/measurement/{}'.format(y_mot)].value).reshape(mapshape)
                Theta_group.create_dataset('x' ,data=x)
                Theta_group.create_dataset('y', data=y)
                Theta_group.create_dataset('Theta', data=Theta)
                Theta_group.create_dataset('XRF', data=fluo_data)
                Theta_group.create_dataset('Monitor', data=mon_data)

                fluo_ori['XRF_norm'][i]=fluo_data/mon_data
                axes['Theta'][i] = Theta
                
        dest_h5.flush()

        print('aligning')
        
        fluo_data=np.asarray(fluo_ori['XRF_norm'])
        fixed_image = int(Theta_pts/2)
        resolutions =  ['4','2','1']
        aligned, shift = ia.elastix_align(fluo_data, mode ='translation', fixed_image_no=fixed_image, NumberOfResolutions = resolutions)

        fluo_aligned.create_dataset(name='XRF_norm',data=aligned, compression='lzf', shuffle=True)

        alignment = merged_data.create_group('alignment')
        alignment.attrs['NXprocess'] = 'NXprocess'
        alignment.create_dataset(name='shift',data=shift)
        alignment_parameters = alignment.create_group('alignment_parameters')
        alignment_parameters.attrs['script'] = ia.__file__
        alignment_parameters.attrs['function'] = 'elastix_align'
        alignment_parameters.attrs['signal'] = fluo_ori.name
        alignment_parameters.attrs['mode'] = 'translation'
        alignment_parameters.attrs['fixed_image_no'] = fixed_image
        alignment_parameters.attrs['NumberOfResolutions'] = resolutions
        
        dest_h5.flush()
    print('written to {}'.format(dest_fname))


def do_fluo_lines_shift(dest_fname, lines_shift_fname=None, verbose=False):
    align_starttime = time.time()
    print('aligning lines in fluodata from \n {}'.format(dest_fname))
    curr_dir = os.path.dirname(dest_fname)

    with h5py.File(dest_fname) as dest_h5:
        
        Theta_list = [[float(key), value.value] for key, value in dest_h5['entry/integrated_files'].items()]
        Theta_list.sort()

        align_g = dest_h5['entry/merged_data/alignment/']
        
        shift = np.asarray(align_g['shift'])

        if type(lines_shift_fname)==str:
            lines_shift = open_data(lines_shift_fname)[0]
            align_g.create_dataset('lines_shift',data=lines_shift)
            align_g.attrs['lines_shift_fname']=lines_shift_fname
            lines_shift_dict = dict(zip([Theta for Theta,filename in  Theta_list],lines_shift))
        elif 'lines_shift' in dest_h5['entry/merged_data/alignment'].keys():
            lines_shift = list(np.asarray(align_g['lines_shift']))
            lines_shift_dict = dict(zip([Theta for Theta,filename in  Theta_list],lines_shift))
        else:
            print('no lines_shift data found in {}'.format(dest_fname))
            return

        fluo_merged = dest_h5['entry/merged_data/fluorescence']
        axes = dest_h5['entry/merged_data/axes']

        fluo_ori_data = np.asarray(fluo_merged['fluo_original/XRF_norm'])
        fluo_shift_aligned = np.zeros_like(fluo_ori_data)
        
        for i,[Theta, fname] in enumerate(Theta_list):
            for j, line in enumerate(fluo_ori_data[i]):
                ndshift(line,lines_shift_dict[Theta][j], output = fluo_shift_aligned[i][j],order=1)

            ndshift(fluo_shift_aligned[i],-shift[i],output=fluo_shift_aligned[i],order=1)
            
        fluo_sa = fluo_merged.create_group('fluo_shift_aligned')
        fluo_sa.create_dataset('XRF_norm',data=fluo_shift_aligned, compression='lzf')
        fluo_sa.attrs['NX_class'] = 'NXdata'
        fluo_sa.attrs['signal'] = 'XRF_norm'
        fluo_sa.attrs['axes'] = ['Theta','y','x']
        fluo_sa['Theta'] = axes['Theta']
        fluo_sa['x'] = axes['x']
        fluo_sa['y'] = axes['y']                      
                                    

def do_align_diffraction_data(dest_fname, no_processes=4, verbose=False):
    parallel_align_diffraction(dest_fname, no_processes, verbose)
    collect_align_diffraction(dest_fname, verbose)


def parallel_align_diffraction(dest_fname, no_processes=4, verbose=False):
    '''
    memory restricted no_process see size of data to be aligned
    '''

    align_starttime = time.time()
    total_datalength = 0
    print('aligning diffraction data from\n{}'.format(dest_fname))
    todo_list = []
    curr_dir = os.path.dirname(dest_fname)
    alignmap_dir = curr_dir+os.path.sep+'diff_aligned'
    
    source_group_subpaths = ['tth_2D/data','raw_data/data']
    
    if os.path.exists(alignmap_dir):
        rmtree(alignmap_dir)
    
    os.mkdir(alignmap_dir)
        
    subdest_fname_tpl = alignmap_dir+os.path.sep+'{}'+os.path.sep+'single_map_{:08d}.h5'

    with h5py.File(dest_fname,'r') as dest_h5:
        
        shift = list(np.asarray(dest_h5['entry/merged_data/alignment/shift']))
        Theta_list = [[float(key), value.value] for key, value in dest_h5['entry/integrated_files'].items()]
        shift_dict = dict(zip([Theta for Theta,filename in  Theta_list],shift))
        print(shift_dict)

        if 'lines_shift' in dest_h5['entry/merged_data/alignment'].keys():

            lines_shift = list(np.asarray(dest_h5['entry/merged_data/alignment/lines_shift']))
            lines_shift_dict = dict(zip([Theta for Theta,filename in  Theta_list],lines_shift))
        else:
            lines_shift_dict = dict(zip([Theta for Theta,filename in  Theta_list],[None]*len(Theta_list)))
             
            
        axes = dest_h5['entry/merged_data/axes']
        
        with h5py.File(Theta_list[0][1]) as first_h5:
            troi_list = first_h5['entry/integrated/'].keys()
            # troi_axes_dict = dict(zip((troi_list),[{}]*len(troi_list)))            
            for troiname in troi_list:
                troi_dir = alignmap_dir+os.path.sep+troiname
                    
                if not os.path.exists(troi_dir):
                    os.mkdir(troi_dir)
                    
                for Theta, fname in Theta_list:
                    # new dest_fname needs to be made to circumvent parrallelism issues
                    subsource_fname = fname
                    subdest_fname = subdest_fname_tpl.format(troiname,int(1000*Theta))

                    subdest_grouppaths = source_group_subpaths
                    troi_grouppath='entry/integrated/{}/'.format(troiname)
                    subsource_grouppaths = [troi_grouppath+x for x in source_group_subpaths]

                    mapshape = (axes['y'].shape[0],axes['x'].shape[0])
                    total_datalength+=mapshape[0]*mapshape[1]

                    todo_list.append([subdest_fname,
                                      subdest_grouppaths,
                                      subsource_fname,
                                      subsource_grouppaths,
                                      Theta,
                                      mapshape,
                                      shift_dict[Theta],
                                      lines_shift_dict[Theta],
                                      verbose])       

    print('setup parallel proccesses to write to {}'.format(alignmap_dir))

    instruction_list = []
    for i,todo in enumerate(todo_list):
        #DEBUG:
        print('todo #{:2d}'.format(i))
        print(todo)
        instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
        instruction_list.append(instruction_fname)

    if no_processes==1:
        for instruction in instruction_list:
            adw.align_data_worker(instruction)
        ## non parrallel version for one dataset and timing:
        #fdw.fit_data_worker(instruction_list[0])
    else:
        pool = Pool(no_processes, worker_init(os.getpid()))
        pool.map_async(adw.align_data_employer,instruction_list)
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
    print('collecting all the parallely processed data in '.format(dest_fname))
    curr_dir = os.path.dirname(dest_fname)
    alignmap_dir = curr_dir+os.path.sep+'diff_aligned/'

    with h5py.File(dest_fname) as dest_h5:
        Theta_list = [[float(key), value.value] for key, value in dest_h5['entry/integrated_files'].items()]
        axes = dest_h5['entry/merged_data/axes']
        
        with h5py.File(Theta_list[0][1],'r') as first_h5:
            troi_list = first_h5['entry/integrated/'].keys()
            for troi in troi_list:
                troi_ax_dest = axes.create_group(troi)
                troi_ax_source = first_h5['entry/integrated/{}/axes'.format(troi)]
                for key, value in troi_ax_source.items():
                    troi_ax_dest.create_dataset(name=key,data=np.asarray(value))


                  
                    
        for troiname in troi_list:
            source_dir = alignmap_dir+troiname+os.path.sep
            fname_list = [os.path.realpath(source_dir + x) for x in os.listdir(source_dir) if x.find('.h5')]
            fname_list.sort()
            troi_merged = dest_h5['entry/merged_data'].create_group(troi_name)
            troi_merged.attrs['NX_class'] = 'NXcollection'
            with h5py.File(fname_list[0],'r') as first_h5:
                merge_datasets = first_h5.keys()
                full_dtypes = [first_h5['{}/data/data'.format(x)].dtype for x in merge_datasets]
                full_shapes = [list(first_h5['{}/data/data'.format(x)].shape) for x in merge_datasets]
                [x.insert(2,len(fname_list)) for x in full_shapes]
                print(full_dtypes)
                print(full_shapes)
                
            for i, dataset_name in enumerate(merge_datasets):
                dataset_path = '{}/data'.format(dataset_name)

                int_merged = troi_merged.create_group(dataset_name)
                int_merged.attrs['NX_class'] = 'NXcollection'
                
                single_maps = int_merged.create_group('single_maps')
                single_maps.attrs['NX_class'] = 'NXprocess'
                
                full_group = int_merged.create_group('all')
                full_ds = full_group.create_dataset('data', shape=full_shapes[i], dtype=full_dtypes[i], compression='lzf')
                full_sum = full_group.create_dataset('data_sum', shape=full_shapes[i][:-2], dtype=full_dtypes[i], compression='lzf')
                full_max = full_group.create_dataset('data_max', shape=full_shapes[i][:-2], dtype=full_dtypes[i], compression='lzf')

          
                for i, subsource_fname in enumerate(fname_list):
                    with h5py.File(subsource_fname,'r') as source_h5:
                        print('getting {} from {}'.format(dataset_name, subsource_fname))
                        Theta = source_h5['raw_data/data/Theta'].value
                        Theta_group = single_maps.create_group(name=str(Theta))
                        Theta_group.attrs['NX_class'] = 'NXdata'
                        Theta_group.attrs['signal'] = 'sum'
                        Theta_group.attrs['source_filename'] = subsource_fname

                        Theta_group.attrs['axes'] = ['x','y']
                        Theta_group['x'] = axes['x']
                        Theta_group['y'] = axes['y']
                        Theta_group['Theta'] = Theta
                        
                        full_ds[:,:,i,:,:] = np.asarray(source_h5[dataset_path+'/data'])
                        full_sum[:,:,i] = np.asarray(source_h5[dataset_path+'/sum'])
                        full_max[:,: ,i] = np.asarray(source_h5[dataset_path+'/max'])
                    # ExternalLink cant find file if its open ?
                    Theta_group['sum'] =  h5py.ExternalLink(subsource_fname, dataset_path+'/sum')
                    Theta_group['max'] =  h5py.ExternalLink(subsource_fname, dataset_path+'/max')
                    Theta_group['data'] = h5py.ExternalLink(subsource_fname, dataset_path+'/data')

        dest_h5.flush()

            
    collect_endtime = time.time()
    collect_time = (collect_endtime - collect_starttime)
    print('='*25)
    print('\ntime taken for collecting all frames = {}'.format(collect_time))
    print('='*25) 
        
def init_h5_file(masterfolder, verbose =False):

    dest_fname = os.path.realpath(masterfolder + os.path.sep + 'merged.h5')
    my_h5_fname_list = find_my_h5_files(masterfolder)
    
    if os.path.exists(dest_fname):
        os.remove(dest_fname)
        print('removing {}'.format(dest_fname))

    print('\nwriting to file')
    print(dest_fname)
        
    with h5py.File(dest_fname) as dest_h5:
        dest_h5.attrs['file_name']        = dest_h5.filename
        dest_h5.attrs['creator']          = os.path.basename(__file__)
        dest_h5.attrs['HDF5_Version']     = h5py.version.hdf5_version
        dest_h5.attrs['NX_class']         = 'NXroot'
        dest_h5.create_group('entry')
        dest_h5['entry'].attrs['NX_class'] = 'NXentry'
        dest_h5.attrs['file_time']  = "T".join(str(datetime.datetime.now()).split())

        merged_data=dest_h5['entry'].create_group('merged_data')
        merged_data.attrs['NX_class'] = 'NXcollection'
        fluo_merged = merged_data.create_group('fluorescence')
        fluo_merged.attrs['NX_class'] = 'NXcollection'
        
        integrated_files = dest_h5['entry'].create_group('integrated_files')
        integrated_files.attrs['NX_class'] = 'NXcollection'

        Theta_list = []
        for i,fname in enumerate(my_h5_fname_list):
            
            # get all Thetas to be able to sort:
            
    
            with h5py.File(fname) as source_h5:
                Theta = source_h5['entry/instrument/positioners/Theta'].value
                Theta_list.append([Theta, fname])
                            
        Theta_list.sort()

        for i,[Theta, fname] in enumerate(Theta_list):
            integrated_files.create_dataset(str(Theta),data=fname)
            if verbose:
                print('fname_{:04d}'.format(i))
                print(fname)
        
                
    
        dest_h5.flush()

    return dest_fname

                
def main():

    # # no_processes = 22

    # ## this is created by read_rois.py:
    masterfolder = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_gpu2/'
    dest_fname = init_h5_file(masterfolder=masterfolder, verbose=True)
    do_fluo_merge(dest_fname, verbose=True)
    # dest_fname = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_gpu2/merged.h5'
    # # ## somehow interactively create data to deglitch:    
    # # ## 'entry/merged_data/alignment/lines_shift
    

    do_fluo_lines_shift(dest_fname, lines_shift_fname='/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/line_shift.dat')
                                    
    # ## first fit then align
    # # do_fit_raw_diffraction_data(dest_fname, no_processes=no_processes, verbose = False)
    # ## first align them fit
    # # do_align_diffraction_data(dest_fname, no_processes=4, verbose=False)
    # one process takes ~6% of gpu2 memory
    parallel_align_diffraction(dest_fname, no_processes=6, verbose=False)
    collect_align_diffraction_data(dest_fname, verbose = True)
        
if __name__ == "__main__":
    main()

