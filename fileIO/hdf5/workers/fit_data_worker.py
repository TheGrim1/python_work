from __future__ import print_function

import numpy as np
import h5py
import os, sys
import pyFAI
from scipy.ndimage.measurements import center_of_mass

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy
import simplecalc.lorentz_fitting as lf
import simplecalc.fitting2d as f2d


def fit_data_employer(pickledargs_fname):
    '''
    gutwrenching way to (almost) completely uncouple the h5 access from the motherprocess
    Can be used to multiprocess.pool control the workers. Allows another layer of parallelisation :D
    '''
    fname =  __file__
    cmd = 'python {} {}'.format(fname, pickledargs_fname)
    
    os_response = os.system(cmd)
    if os_response >1:
        raise ValueError('in {}\nos.system() has responded with errorcode {} in process {}'.format(fname, os_response, os.getpid))
    
def fit_subcontract(args):
    '''
    try to fit no_peaks lorenzians to tth and chi and the same number of 2d gaussians to d2_data. i is passed for indexing
    '''
    i = args[0]
    tth = args[1]
    chi = args[2]
    d2_data = args[3]
    no_peaks = args[4]
    
    
    tth_result = lf.do_iterative_variable_lorentzbkg_pipeline(tth,
                                                              nopeaks=no_peaks,
                                                              minwidth=2,
                                                              verbose=False)[0]

    chi_result = lf.do_iterative_variable_lorentzbkg_pipeline(chi,
                                                              nopeaks=no_peaks,
                                                              minwidth=2,
                                                              verbose=False)[0]

    tth_com = np.interp(center_of_mass(tth[1]),np.arange(len(tth[0])),tth[0])
    chi_com = np.interp(center_of_mass(chi[1]),np.arange(len(chi[0])),chi[0])

    # need to sort and trim to no_peaks
    fit_result = f2d.do_multiple_gauss2d_fit(d2_data, d2_chi, d2_tth, force_positive=True)
    found_peaks = len(fit_result)/6
    fit_result = fit_result.reshape(found_peaks,6)

    areas = []
    for j,peak in enumerate(fit_result):
        # print(peak)
        if peak[2] * peak[3] > 500: # peak is too wide
            fit_result[i]*=0
        areas.append([peak[5],j]) # list for sorting
    areas.sort()

    d2_result = np.empty(dtype=d2_data.dtype, shape=(no_peaks,6))
    for k in range(no_peaks):
        if k<found_peaks:
            d2_result[k] = fit_result[areas[k][1]]
        else:
            d2_result[k] = np.zeros(6)

    return i, tth_result, chi_result, tth_com, chi_com, d2_result
    
def fit_data_worker(pickledargs_fname):
    '''
    interates fit routine over source_name[source_grouppath][dataset_path] and stores results in target_fname[target_group][results_path]
    these datasets are created here
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)

    print(os.getpid()) 
    print(unpickled_args)
   
    target_fname = unpickled_args[0]
    target_grouppath = unpickled_args[1]
    source_fname = unpickled_args[2]
    source_grouppath = unpickled_args[3]
    Theta = unpickled_args[4]
    mapshape = unpickled_args[5]
    no_peaks = unpickled_args[6]
    verbose = unpickled_args[7]
        
    if verbose:
        print('='*25)
        print('process {} is doing'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)

    with h5py.File(target_fname) as target_file:
        target_group = target_file.create_group(target_grouppath)
        
        with h5py.File(source_fname) as source_file:
            
            source_group = source_file[source_grouppath]
            dtype = source_group['tth_radial/I'].dtype

            target_group.create_dataset('Theta', data=Theta)
            target_group.create_dataset('sum', data=np.asarray(source_group['tth_radial/I']).sum(1).reshape(mapshape))
            target_group.create_dataset('max', data=np.asarray(source_group['tth_radial/I']).max(1).reshape(mapshape))
            target_group.create_dataset('chi_com', dtype=dtype, shape = mapshape)
            target_group.create_dataset('tth_com', dtype=dtype, shape = mapshape)
            target_group.create_dataset('chi_fit', dtype=dtype, shape = (mapshape[0],mapshape[1],no_peaks,3))            
            target_group.create_dataset('tth_fit', dtype=dtype, shape = (mapshape[0],mapshape[1],no_peaks,3))
            target_group.create_dataset('2d_fit', dtype=dtype, shape = (mapshape[0],mapshape[1],no_peaks,6))

            no_frames = mapshape[0]*mapshape[1]
            tth_result = np.empty(dtype=dtype, shape=(no_frames,no_peaks,3))
            chi_result = np.empty(dtype=dtype, shape=(no_frames,no_peaks,3))
            d2_result = np.empty(dtype=dtype, shape=(no_frames,no_peaks,6))

            tth_axis = np.asarray(source_group['axes/tth'])
            chi_axis = np.asarray(source_group['axes/chi_azim'])

            tth_com = np.empty(dtype=dtype, shape=(no_frames))
            chi_com = np.empty(dtype=dtype, shape=(no_frames))

            d2_tth, d2_chi = np.meshgrid(tth[0],chi[0])
            d2_data = np.empty(dtype=dtype, shape=(d2_chi.shape))

            # print('data_fitter for {} gets:'.format(source_grouppath))
            # print(source_fname)
            # print(tth.shape)
            # print(tth.dtype)
                
            todo_list = []
            for i in range(no_frames):
                tth_data = np.asarray(source_group['tth_radial/I'][i])
                chi_data = np.asarray(source_group['chi_azimuthal/I'][i])
                
                tth = np.array(dtype=dtype, data=np.stack((tth_axis,tth_data)))
                chi = np.array(dtype=dtype, data=np.stack((chi_axis,chi_data)))                
                d2_data = np.asarray(source_group['tth_2D/data'][i])

                todo = []
                todo.append(i)
                todo.append(tth)
                todo.append(chi)
                todo.append(d2_data)
                todo.append(no_peaks)
                todo_list.append(todo)                

        pool = Pool(noprocesses = 20)
        result = pool.map_async(fit_subcontract, todo_list)
        pool.close()
        pool.join()

        for i, tth_result_i, chi_result_i, tth_com_i, chi_com_i, d2_result_i in result.get():
            tth_result[i] =  tth_result_i
            chi_result[i] = chi_result_i
            tth_com[i] = tth_com_i
            chi_com[i] = chi_com_i
            d2_result[i] = d2_result_i
    
        target_group['tth_fit'][:] = tth_result.reshape(mapshape[0],mapshape[1],no_peaks,3)
        target_group['chi_fit'][:] = chi_result.reshape(mapshape[0],mapshape[1],no_peaks,3)
        target_group['tth_com'][:] = np.asarray(tth_com).reshape(mapshape)
        target_group['chi_com'][:] = np.asarray(chi_com).reshape(mapshape)
        target_group['2d_fit'][:] = np.asarray(d2_result).reshape(mapshape[0],mapshape[1],no_peaks,6)
        
        target_file.flush()
            
    if verbose:
        print('process {} is done'.format(os.getpid()))
        print('='*25)


if __name__=='__main__':
    ''' 
    This is used by the local function integrate_troi_employer(pickledargs_fname),
    DO NOT CHANGE
    '''
    if len(sys.argv)!=2:
        print('usage : python integrate_troi_worker <pickled_instruction_list_fname>')
    pickledargs_fname = sys.argv[1]
    fit_data_worker(pickledargs_fname)
                        
