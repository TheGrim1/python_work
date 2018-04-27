from __future__ import print_function

import numpy as np
import h5py
import os, sys
import pyFAI
import scipy.ndimage.measurements as meas

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy
import simplecalc.lorentz_fitting as lf


def fit_data_employer(pickledargs_fname):
    '''
    gutwrenching way to (almost) completely uncouple the h5 access from the motherprocess
    Can be used to multiprocess.pool control the workers.
    '''
    fname =  __file__
    cmd = 'python {} {}'.format(fname, pickledargs_fname)
    
    os_response = os.system(cmd)
    if os_response >1:
        raise ValueError('in {}\nos.system() has responded with errorcode {} in process {}'.format(fname, os_response, os.getpid()))
    

def fit_data_worker(pickledargs_fname):
    '''
    interates fit routine over source_name[source_grouppath][dataset_path] and stores results in target_fname[target_group][results_path]
    these datasets have to already exist with the right shape and dtype and no compression!
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

            no_frames = mapshape[0]*mapshape[1]
            tth_result = np.empty(dtype=dtype, shape=(no_frames,no_peaks,3))
            chi_result = np.empty(dtype=dtype, shape=(no_frames,no_peaks,3))

            tth_pts = source_group['axes/tth'].shape[0]
            chi_pts = source_group['axes/chi_azim'].shape[0]
            
            tth = np.empty(dtype=dtype, shape=(2,tth_pts))
            tth[0] = np.asarray(source_group['axes/tth'])

            chi = np.empty(dtype=dtype, shape=(2,chi_pts))
            chi[0] = np.asarray(source_group['axes/chi_azim'])

            tth_com = np.empty(dtype=dtype, shape=(no_frames))
            chi_com = np.empty(dtype=dtype, shape=(no_frames))

            # print('data_fitter for {} gets:'.format(source_grouppath))
            # print(source_fname)
            # print(tth.shape)
            # print(tth.dtype)
                
            
            for i in range(no_frames):
                tth[1] = np.asarray(source_group['tth_radial/I'][i])
                chi[1] = np.asarray(source_group['chi_azimuthal/I'][i])


                tth_result[i] = lf.do_iterative_variable_lorentzbkg_pipeline(tth,
                                                                             nopeaks=no_peaks,
                                                                             minwidth=2,
                                                                             verbose=verbose)[0]
                
                chi_result[i] = lf.do_iterative_variable_lorentzbkg_pipeline(chi,
                                                                             nopeaks=no_peaks,
                                                                             minwidth=2,
                                                                             verbose=verbose)[0]
                               
                tth_com[i] = np.interp(meas.center_of_mass(tth[1]),np.arange(tth_pts),tth[0])
                chi_com[i] = np.interp(meas.center_of_mass(chi[1]),np.arange(chi_pts),chi[0])
                
        target_group['chi_fit'][:] = chi_result.reshape(mapshape[0],mapshape[1],no_peaks,3)
        target_group['tth_fit'][:] = tth_result.reshape(mapshape[0],mapshape[1],no_peaks,3)
        target_group['tth_com'][:] = np.asarray(tth_com).reshape(mapshape)
        target_group['chi_com'][:] = np.asarray(chi_com).reshape(mapshape)
        
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
                        
