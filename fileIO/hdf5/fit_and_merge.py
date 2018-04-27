import h5py
import numpy as np
import sys, os
import datetime
import time
from multiprocessing import Pool
from scipy.ndimage import shift as ndshift
from shutil import rmtree
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

from simplecalc.slicing import troi_to_corners
from simplecalc.image_align import do_shift
import simplecalc.image_align_elastix as ia
import pythonmisc.pickle_utils as pu
import fileIO.hdf5.workers.fit_data_worker as fdw

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


def process_integrated_dataset(args):
    pass

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
            axes.attrs['NXclass'] = 'NXcollection'
            axes.create_dataset('Theta', dtype= y.dtype, shape = (Theta_pts,))
            axes.create_dataset('x' ,data=x)
            axes.create_dataset('y', data=y)
        
            fluo_ori = fluo_merged.create_group('fluo_original')
            fluo_ori.attrs['NXclass'] = 'NXdata'
            fluo_ori.attrs['signal'] = 'XRF_norm'
            fluo_ori.attrs['axes'] = ['Theta','y','x']
            fluo_ori['Theta'] = axes['Theta']
            fluo_ori['x'] = axes['x']
            fluo_ori['y'] = axes['y']
            fluo_ori.create_dataset(name='XRF_norm', dtype=y.dtype, shape=(Theta_pts, y_pts, x_pts), compression='lzf', shuffle=True)

            fluo_aligned = fluo_merged.create_group('fluo_aligned')
            fluo_aligned.attrs['NXclass'] = 'NXdata'
            fluo_aligned.attrs['signal'] = 'XRF_norm'
            fluo_aligned.attrs['axes'] = ['Theta','y','x']
            fluo_aligned['Theta'] = axes['Theta']
            fluo_aligned['x'] = axes['x']
            fluo_aligned['y'] = axes['y']

            single_maps = fluo_merged.create_group('single_maps')
            single_maps.attrs['NXclass'] = 'NXprocess'
    
        for i,[Theta, fname] in enumerate(Theta_list):
            with h5py.File(fname) as source_h5:

                print('reading no {} of {}'.format(i+1,Theta_pts))
                print('Theta {}, file {}'.format(Theta,fname))
                
                fluo_data = np.asarray(source_h5['entry/instrument/measurement/{}'.format(fluo_counter)].value).reshape(mapshape)
                mon_data = np.asarray(source_h5['entry/instrument/measurement/{}'.format('Monitor')].value).reshape(mapshape)

                Theta_group = single_maps.create_group(name=str(Theta))
                Theta_group.attrs['NXclass'] = 'NXdata'
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

def do_fit_diffraction_data(dest_fname,  no_processes=11, verbose=False):

    fit_starttime = time.time()
    total_datalength = 0
    no_peaks = 4
    print('fitting diffraction data with {} processes'.format(no_processes))
    todo_list=[]
    print('created directory for single maps')
    curr_dir = os.path.dirname(dest_fname)
    diffmap_dir = curr_dir+os.path.sep+'temp_diff_maps'

    if os.path.exists(diffmap_dir):
        rmtree(diffmap_dir)
    
    os.mkdir(diffmap_dir)
        
    subdest_fname_tpl = diffmap_dir+os.path.sep+'{}'+os.path.sep+'single_map_{:08d}.h5'
          
    with h5py.File(dest_fname) as dest_h5:
        
        Theta_list = [[float(key), value.value] for key, value in dest_h5['entry/integrated_files'].items()]
        diff_merged = dest_h5['entry/merged_data'].create_group('diffraction')
        diff_merged.attrs['NXclass'] = 'NXcollection'
        axes = dest_h5['entry/merged_data/axes']
        axes.create_dataset(name='peak_number', data=np.arange(no_peaks))
        axes.create_dataset(name='peak_parameter', data=['a','mu','sig'])
        
        
        with h5py.File(Theta_list[0][1]) as first_h5:
            troi_list = first_h5['entry/integrated/'].keys()

            for troiname in troi_list:
                troi_merged = diff_merged.create_group(troiname)
                single_maps = troi_merged.create_group('single_maps')
                troi_dir = diffmap_dir+os.path.sep+troiname
                if not os.path.exists(troi_dir):
                    os.mkdir(troi_dir)
                single_maps.attrs['NXclass'] = 'NXprocess'
                


                for Theta, fname in Theta_list:
                    # new dest_fname needs to be made to circumvent parrallelism issues

                    subdest_fname = subdest_fname_tpl.format(troiname,int(1000*Theta))
                    subdest_grouppath = 'data'
                    
                    Theta_group = single_maps.create_group(name=str(Theta))
                    Theta_group.attrs['NXclass'] = 'NXdata'
                    Theta_group.attrs['signal'] = 'sum'
                    source_fname = fname
                    Theta_group.attrs['source_filename'] = source_fname
                    source_grouppath = 'entry/integrated/{}/'.format(troiname)
                    Theta_group.attrs['source_h5_path'] = source_grouppath
                    Theta_group.attrs['axes'] = ['x','y','peak_number','peak_parameter']
                    Theta_group['x'] = axes['x']
                    Theta_group['y'] = axes['y']
                    Theta_group['peak_number'] = axes['peak_number']
                    Theta_group['peak_parameter'] = axes['peak_parameter']
                    Theta_group['Theta'] = axes['Theta']
                    dtype = axes['y'].dtype
                    mapshape = (axes['y'].shape[0],axes['x'].shape[0])

                    Theta_group.create_dataset('sum', dtype=dtype, shape = mapshape)
                    Theta_group.create_dataset('max', dtype=dtype, shape = mapshape)
                    Theta_group.create_dataset('chi_com', dtype=dtype, shape = mapshape)
                    Theta_group.create_dataset('tth_com', dtype=dtype, shape = mapshape)
                    Theta_group.create_dataset('chi_fit', dtype=dtype, shape = (mapshape[0],mapshape[1],no_peaks,3))            
                    Theta_group.create_dataset('tth_fit', dtype=dtype, shape = (mapshape[0],mapshape[1],no_peaks,3))
                    total_datalength+=mapshape[0]*mapshape[1]

                    todo_list.append([subdest_fname,
                                      subdest_grouppath,
                                      source_fname,
                                      source_grouppath,
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
            fdw.fit_data_worker(instruction)
        ## non parrallel version for one dataset and timing:
        #fdw.fit_data_worker(instruction_list[0])
    else:
        pool = Pool(processes=no_processes)
        pool.map_async(fdw.fit_data_employer,instruction_list)
        pool.close()
        pool.join()


    

    print('collecting all the parallely processed data in '.format(dest_fname))
    with h5py.File(dest_fname) as dest_h5:
        for troiname in troi_list:
            source_dir = diffmap_dir+os.path.sep+troiname
            fname_list = [source_dir+os.path.sep+x for x in os.listdir(source_dir) if x.find('.h5')]
            fname_list.sort()
            for i, source_fname in enumerate(fname_list):
                with h5py.File(source_fname) as source_h5:
                    Theta = source_h5['data/Theta']
                    Theta_group = dest_h5['entry/merged_data/diffraction/{}/single_maps/{}'.format(troiname,Theta)]
                    Theta_group['sum'][:] = np.asarray(source_h5['data/sum'])
                    Theta_group['max'][:] = np.asarray(source_h5['data/max'])
                    Theta_group['chi_com'][:] = np.asarray(source_h5['data/chi_com'])
                    Theta_group['tth_com'][:] = np.asarray(source_h5['data/tth_com'])
                    Theta_group['tth_fit'][:] = np.asarray(source_h5['data/tth_fit'])
                    Theta_group['chi_fit'][:] = np.asarray(source_h5['data/chi_fit'])
        dest_h5.flush()
    
    fit_endtime = time.time()
    fit_time = (fit_endtime - fit_starttime)
    print('='*25)
    print('\ntime taken for fitting of {} frames = {}'.format(total_datalength, fit_time))
    print(' = {} Hz\n'.format(total_datalength/fit_time))
    print('='*25) 
    
def do_merge_diffraction_data(dest_file, verbose=False):
    
    with h5py.File(dest_fname) as dest_h5:

        troi_list = dest_h5['entry/merged_data/diffraction'].keys()
        shift = list(np.asarray(dest_h5['entry/merged_data/shift/shift']))
        diff_merged = dest_h5['entry/merged_data/diffraction'] 

        axes  = dest_h5['entry/merged_data/axes']
        Theta_pts = axes['Theta'].shape[0]
        y_pts = axes['y'].shape[0]
        x_pts = axes['x'].shape[0]
        no_peaks = axes['peak_number'].shape[0]
        no_pp = axes['peak_parameters'].shape[0]
        dtype = axes['x'].dtype
        mapshape = (y_pts,x_pts)
        datashape = (Theta_pts, y_pts, x_pts)
       
        for troiname in troi_list:    

            troi_group = diff_merged[troiname]
            first_map = troi_group['single_maps'].values()[0]
            no_peaks = first_map['chi_fit'].shape[2]
            
            diff_ori = troi_group.create_group('diff_original')
            diff_ori.attrs['NXclass'] = 'NXdata'
            diff_ori.attrs['signal'] = 'sum'
            diff_ori.attrs['axes'] = ['Theta','y','x']
            diff_ori['Theta'] = axes['Theta']
            diff_ori['x'] = axes['x']
            diff_ori['y'] = axes['y']

            diff_ali = troi_group.create_group('diff_aligned')
            diff_ali.attrs['NXclass'] = 'NXdata'
            diff_ali.attrs['signal'] = 'sum'
            diff_ali.attrs['axes'] = ['Theta','y','x']
            diff_ali['Theta'] = axes['Theta']
            diff_ali['x'] = axes['x']
            diff_ali['y'] = axes['y']

            
            diff_ori.create_dataset('sum', dtype=dtype, shape = datashape)
            diff_ori.create_dataset('max', dtype=dtype, shape = datashape)
            diff_ori.create_dataset('chi_com', dtype=dtype, shape = datashape)
            diff_ori.create_dataset('tth_com', dtype=dtype, shape = datashape)
            diff_ori.create_dataset('chi_fit', dtype=dtype, shape = (datashape[0], datashape[1], datashape[2], no_peaks,3))            
            diff_ori.create_dataset('tth_fit', dtype=dtype, shape = (datashape[0], datashape[1], datashape[2], no_peaks,3))

            diff_ali.create_dataset('sum', dtype=dtype, shape = datashape)
            diff_ali.create_dataset('max', dtype=dtype, shape = datashape)
            diff_ali.create_dataset('chi_com', dtype=dtype, shape = datashape)
            diff_ali.create_dataset('tth_com', dtype=dtype, shape = datashape)
            diff_ali.create_dataset('chi_fit', dtype=dtype, shape = (datashape[0], datashape[1], datashape[2], no_peaks,3))            
            diff_ali.create_dataset('tth_fit', dtype=dtype, shape = (datashape[0], datashape[1], datashape[2], no_peaks,3))
            

            Theta_list = list(axes[Theta])
            
            for i, Theta in enumerate(Theta_list):
                Theta_group = troi_group['single_maps/{}'.format(Theta)]
                diff_ori['sum'][i] = np.asarray(Theta_group['sum'])
                diff_ori['max'][i] = np.asarray(Theta_group['max'])
                diff_ori['chi_com'][i] = np.asarray(Theta_group['chi_com'])
                diff_ori['tth_com'][i] = np.asarray(Theta_group['tth_com'])
                diff_ori['chi_fit'][i] = np.asarray(Theta_group['chi_fit'])
                diff_ori['tth_fit'][i] = np.asarray(Theta_group['tth_fit'])
                
                diff_ali['sum'][i] = ndshift(np.asarray(Theta_group['sum']),shift[i])
                diff_ali['max'][i] = ndshift(np.asarray(Theta_group['max']),shift[i])
                diff_ali['chi_com'][i] = ndshift(np.asarray(Theta_group['chi_com']),shift[i])
                diff_ali['tth_com'][i] = ndshift(np.asarray(Theta_group['tth_com']),shift[i])

                for pn in range(no_peaks):
                    for pp in range(no_parameters):
                        diff_ali['chi_fit'][i,:,:,pn,pp] = ndshift(np.asarray(Theta_group['chi_fit'][:,:,pn,pp]),shift[i])
                        diff_ali['tth_fit'][i,:,:,pn,pp] = ndshift(np.asarray(Theta_group['tth_fit'][:,:,pn,pp]),shift[i])
                        
        dest_h5.flush()
        
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

    no_processes = 11
    masterfolder = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_gpu2/'
    dest_fname = init_h5_file(masterfolder=masterfolder, verbose=True)
    do_fluo_merge(dest_fname, verbose=True)
    dest_fname = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_1/merged.h5'
    
    do_fit_diffraction_data(dest_fname, no_processes=no_processes, verbose = False)
    # do_q_merge(dest_fname)
        
if __name__ == "__main__":
    main()

