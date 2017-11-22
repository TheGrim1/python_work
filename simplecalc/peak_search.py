import sys,os
import numpy as np
import scipy.ndimage as nd
import fabio
import cv2
from multiprocessing import Pool
import h5py
import gc
import time

# local imports
path_list = os.path.realpath(os.path.dirname(__file__)).split(os.path.sep)
importpath_list = []
if 'skript' in path_list:
    for folder in path_list:
        importpath_list.append(folder)
        if folder == 'skript':
            break
importpath = os.path.sep.join(importpath_list)
sys.path.append(importpath)


from fileIO.hdf5.open_h5 import open_h5
from fileIO.hdf5.save_h5 import save_h5
from fileIO.hdf5.save_h5 import merge_h5

from simplecalc.filter_peaks_replace_parallel import remove_close_peaks  

def get_backgroundmask(data_fname,threshold,frame_list=None):
    '''
    use a minprojection to make a mask where min(data)>threshold
    '''
    data = open_h5(data_fname,framelist=frame_list)
    minproj = np.zeros(shape=data[0].shape)
    minproj += threshold
    for i, frame in enumerate(data):
        print'projecting frame %s of %s'%(i,data.shape[0])
        minproj = np.where(frame<minproj, frame, minproj)

    mask = np.where(minproj>=threshold,1,0)
    
    return mask, minproj


def decide_on_single_frame(args):

    data_fname,iframe,mask,FILTER_PARAMETERS,verbose = args
    
    NFRAMES = FILTER_PARAMETERS['NFRAMES']
    GAUSSIAN_PEAK_FILTER_SIGMA = FILTER_PARAMETERS['GAUSSIAN_PEAK_FILTER_SIGMA']
    GAUSSIAN_BACKGROUND_FILTER_SIGMA = FILTER_PARAMETERS['GAUSSIAN_BACKGROUND_FILTER_SIGMA']
    MAX_PXL_PER_PEAK = FILTER_PARAMETERS['MAX_PXL_PER_PEAK']
    MIN_NPEAKS = FILTER_PARAMETERS['MIN_NPEAKS']
    THRESHOLD = FILTER_PARAMETERS['THRESHOLD']
    MIN_DISTANCE = FILTER_PARAMETERS['MIN_DISTANCE']
    pid = os.getpid()
    if verbose:
        print 'filtering frame %s of %s in process %s' %(iframe, NFRAMES, pid)


    data = open_h5(data_fname,[iframe])[0]

    # print 'read data'
    # mask
    frame = np.where(mask,0,data)
    #print 'frame.shape = ', frame.shape
    # gaussian peak filter 
    peak_filtered = nd.gaussian_filter(frame,GAUSSIAN_PEAK_FILTER_SIGMA)
    bkg_filtered = nd.gaussian_filter(frame,GAUSSIAN_BACKGROUND_FILTER_SIGMA)
    filtered = peak_filtered - bkg_filtered
    
    # thresholding
    filtered = np.where(filtered>THRESHOLD,1,0)
    
    # counting and labeling peaks
    npeaks, labeled_image = cv2.connectedComponents(np.asarray(filtered,dtype=np.int8))
    npeaks = npeaks -1 # dont count background, label = 0
    
    # print 'labled data'
    # deciding which to keep
    if npeaks > MIN_NPEAKS:
        # print 'preliminarily found enough peaks'
        
        for j in range(1,npeaks):
            if np.sum(np.where(labeled_image==j,1,0)) > MAX_PXL_PER_PEAK:
                if verbose:
                    print 'peaks too large in frame ', iframe
                return [iframe,2]
    else:
        if verbose:
            print 'rejected frame ' , iframe
        return [iframe,0]

        
    # removing peaks that are closer than MIN_DISTACE from a larger peak:
    peaklist = []
    for j in range(1,npeaks):
        (x_pxls, y_pxls) = np.where(labeled_image==j)
        x_com = np.mean(x_pxls)
        y_com = np.mean(y_pxls)
        n_pxl = np.sum(np.where(labeled_image==j,1,0))
        peaklist.append([x_com,y_com,n_pxl])
        
    peaklist_after = remove_close_peaks(np.asarray(peaklist), MIN_DISTANCE)
    npeaks_far_apart = len([x for x in peaklist_after if x[0]>0])
    npeaks_removed = (npeaks - npeaks_far_apart)
    if verbose:
        print 'removed %s peaks that were close together' % npeaks_removed
        # print 'peaks before:'
        # print peaklist
        # print 'peaks after:'
        # print peaklist_after

    
    if npeaks_far_apart > MIN_NPEAKS:
        # print 'finally found enough peaks'
        if verbose:
            print 'selected frame ', iframe
        return [iframe,1]
    else:
        if verbose :
            print 'rejected frame ' , iframe
        return [iframe,0]





def filter_for_peaks_per_file(args):
    data_fname, selected_fname, mask, FILTER_PARAMETERS, parallel, verbose= args

    NPROCESSES=20

    f = h5py.File(data_fname, 'r')
    print 'reading file ',data_fname
    NFRAMES = f['entry/data/data'].shape[0]    
    FILTER_PARAMETERS['NFRAMES'] = NFRAMES
    f.close()
    
    task_list_frames = [[data_fname, iframe, mask, FILTER_PARAMETERS,verbose] for iframe in range(NFRAMES)]
    
    frame_descision_list = []

    if parallel:
    # parallel implementation
        pool = Pool(processes=NPROCESSES)
        bla = pool.map_async(decide_on_single_frame, task_list_frames)
        bla.wait()
        frame_descision_list = bla.get()
        pool.close()
        pool.join()
    else:
        for task in task_list_frames:
            frame_descision_list.append(decide_on_single_frame(task))
            
    
    selected_frames = [x[0] for x in frame_descision_list if x[1] ==1]
    if len(selected_frames)==0:
        print 'no frames found to match critea'
    else:
        print 'found frames ', selected_frames
    
        if os.path.exists(selected_fname):
            print 'overwriting :'
            print selected_fname
            os.remove(selected_fname)
        save_h5(open_h5(data_fname,selected_frames), selected_fname)
    print 'finished with ', data_fname
    
def filter_many_parallel(data_tpl='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/DATA/AUTO-TRANSFER/eiger4/COF505_dry_heflush_270_data_%06d.h5',
                         save_tpl='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/filter_results/COF505_dry_heflush_270_peaksearchfilterd_%06d.h5',
                         mask_fname ='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/calib1_detx0/mask_Eiger_center.edf',
                         parallel = True,
                         verbose=2):


    mask = np.asarray(fabio.open(mask_fname).data,dtype = np.int8)
    FILTER_PARAMETERS = {}
    FILTER_PARAMETERS['GAUSSIAN_PEAK_FILTER_SIGMA'] = 3
    FILTER_PARAMETERS['GAUSSIAN_BACKGROUND_FILTER_SIGMA'] = 5*FILTER_PARAMETERS['GAUSSIAN_PEAK_FILTER_SIGMA']
    FILTER_PARAMETERS['MAX_PXL_PER_PEAK'] = 500
    FILTER_PARAMETERS['MIN_NPEAKS'] = 10
    FILTER_PARAMETERS['MIN_DISTANCE'] = 100
    FILTER_PARAMETERS['THRESHOLD'] = 2
    NPROCESSES = 20
    if parallel:
        subprocess_parallel = False
    else:
        subprocess_parallel = True

        
    i = 1
    d_name=data_tpl%i
    s_name = save_tpl %i       

    print d_name
    time.sleep(1)
    task_list_files = []
    #while i <20:
    while os.path.exists(d_name):
        print 'adding file to tasklist ',d_name
        task_list_files.append([d_name, s_name, mask, FILTER_PARAMETERS,subprocess_parallel, verbose])
        i+=1
        d_name=data_tpl%i
        s_name = save_tpl %i
        

                         
    if parallel:
    # parallel implementation:

            pool = Pool(processes=NPROCESSES)
            pool.map(filter_for_peaks_per_file, task_list_files)
            pool.close()
            pool.join()
    else:
        # or single thread:
        for task in task_list_files:
            filter_for_peaks_per_file(task)

          
def test():
    data_fname = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/DATA/AUTO-TRANSFER/eiger4/COF505_dry_heflush_270_data_000001.h5'
    tmp_fname = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/temp.tmp'
    selected_fname = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/COF505_dry_heflushe_270_peaksearchfilterd_000001.h5'

    mask_fname ='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/calib1_detx0/mask_Eiger_center.edf'
    mask = np.asarray(fabio.open(mask_fname).data,dtype = np.int8)

    
    GAUSSIAN_PEAK_FILTER_SIGMA = 3
    GAUSSIAN_BACKGROUND_FILTER_SIGMA = 3*GAUSSIAN_PEAK_FILTER_SIGMA 
    THRESHOLD = 1
    MIN_NPEAKS = 20
    MAX_PXL_PER_PEAK = 500

    FILTER_PARAMETERS = {}
    f = h5py.File(data_fname, 'r')
    all_data_shape= list(f['entry/data/data'].shape)
    # do only fames up to :
    all_data_shape[0] = 100
    NFRAMES = all_data_shape[0]    
    FILTER_PARAMETERS['NFRAMES'] = NFRAMES
    f.close()
    FILTER_PARAMETERS['GAUSSIAN_PEAK_FILTER_SIGMA'] = GAUSSIAN_PEAK_FILTER_SIGMA
    FILTER_PARAMETERS['GAUSSIAN_BACKGROUND_FILTER_SIGMA'] = GAUSSIAN_BACKGROUND_FILTER_SIGMA
    FILTER_PARAMETERS['MAX_PXL_PER_PEAK'] = MAX_PXL_PER_PEAK
    FILTER_PARAMETERS['MIN_NPEAKS'] = MIN_NPEAKS
    FILTER_PARAMETERS['THRESHOLD'] = THRESHOLD

    
    # frame_list = range(1000)
    
    # change to do this frame by frame later
    # data = open_h5(data_fname,frame_list)

    selected_frame_list = []
    toolarge_frame_list = []
    
    all_filtered = np.memmap(filename=tmp_fname,
                             mode = 'w+',
                             shape = tuple(all_data_shape),
                             dtype = np.int16)

    for i in range(NFRAMES):

        
        data = open_h5(data_fname,[i])[0]
        print 'filtering frame %s of %s' %(i, NFRAMES)

        # mask
        frame = np.where(mask,0,data)
        #print 'frame.shape = ', frame.shape
        # gaussian filter 
        filtered = nd.gaussian_filter(frame,GAUSSIAN_PEAK_FILTER_SIGMA)

        # thresholding
        filtered = np.where(filtered>THRESHOLD,1,0)

        # counting and labeling peaks
        npeaks, labeled_image = cv2.connectedComponents(np.asarray(filtered,dtype=np.int8))

        
        # deciding which to keep
        if npeaks > MIN_NPEAKS:
            print 'found enough peaks'
            take_this_frame = True
            for j in range(1,npeaks):
                npxl_j = np.where(labeled_image==j)[0].shape[0]
                if npxl_j > MAX_PXL_PER_PEAK:
                    take_this_frame = False
                    print 'found %s pxl in peak %s'%(npxl_j, j)
                    toolarge_frame_list.append(i)
                    break
            if take_this_frame:
                print 'selected this frame'
                selected_frame_list.append(i)
        all_filtered[i] = filtered

    selected=all_filtered
        
    if os.path.exists(selected_fname):
        print 'overwriting :'
        print selected_fname
        os.remove(selected_fname)
    save_h5(selected, selected_fname)
        
    # neccessary cleanup for memmap
    memmap_variable = all_filtered
    if type(memmap_variable) == np.core.memmap:
        print 'cleaning up memmap'
        memmap_tmp_fname = memmap_variable.filename
        del memmap_variable
        gc.collect()
        os.remove(memmap_tmp_fname)


        

if __name__=='__main__':

    
    # # final search in the COF505 data to look for nicely diffracting frames
    # FILTER_PARAMETERS['GAUSSIAN_PEAK_FILTER_SIGMA'] = 3
    # FILTER_PARAMETERS['GAUSSIAN_BACKGROUND_FILTER_SIGMA'] = 5*FILTER_PARAMETERS['GAUSSIAN_PEAK_FILTER_SIGMA']
    # FILTER_PARAMETERS['MAX_PXL_PER_PEAK'] = 500
    # FILTER_PARAMETERS['MIN_NPEAKS'] = 10
    # FILTER_PARAMETERS['MIN_DISTANCE'] = 100
    # FILTER_PARAMETERS['THRESHOLD'] = 2

    
    data_tpl='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/DATA/AUTO-TRANSFER/eiger4/COF505_dry_heflushb_332_data_%06d.h5'
    save_tpl='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/filter_results/COF505_dry_heflushb_332_superfiltered3_%06d.h5'
    mask_fname ='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/calib1_detx0/mask_Eiger_largecenter.edf'
    print data_tpl
    print save_tpl
    time.sleep(1)
    filter_many_parallel(data_tpl=data_tpl,save_tpl=save_tpl,mask_fname=mask_fname,verbose=True,parallel=True)


    data_tpl='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/DATA/AUTO-TRANSFER/eiger4/COF505_dry_heflush_270_data_%06d.h5'
    save_tpl='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/filter_results/COF505_dry_heflush_270_superfiltered3_%06d.h5'
    mask_fname ='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/calib1_detx0/mask_Eiger_largecenter.edf'

    print data_tpl
    print save_tpl
    time.sleep(1)
    
    filter_many_parallel(data_tpl=data_tpl,save_tpl=save_tpl,mask_fname=mask_fname,verbose=True,parallel=True)

    merge_h5(search_phrase='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/filter_results/*superfiltered3*',
             group='entry/data/data',
             save_fname='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/filter_results/COF505_dry_heflushb_332_allsuperfiltered_000001.h5',
             tmp_fname='/data/id13/inhouse8/THEDATA_I8_1/temp.tmp',
             verbose=True)

