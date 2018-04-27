'''
same boilerplate as skripst/fileIO/hdf5/integrate_data_worker.py
'''


from __future__ import print_function

import numpy as np
import h5py
import os, sys
import pyFAI

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy
from fileIO.edf import save_edf

def write_data_to_edf_employer(pickledargs_fname):
    '''
    gutwrenching way to completely uncouple the h5 access from the motherprocess
    Can be used to multiprocess.pool control the workers.
    '''
    fname =  __file__
    cmd = 'python {} {}'.format(fname, pickledargs_fname)

    # import inspect
    # linno = inspect.currentframe().f_lineno
    # print('DEBUG:\nin ' + __file__ + '\nline '+str(linno))
    # print(cmd)

    
    os_response = os.system(cmd)
    if os_response >1:
        raise ValueError('in {}\nos.system() has responded with errorcode {} in process {}'.format(fname, os_response, os.getpid))
    

def write_data_to_edf_worker(pickledargs_fname):
    '''
    interates troi into target_fname[target_group][XXX/data][target_index] from source_name[source_datasetpath][source_index][troi]
    with XXX = 'q_integration_1D','q_integration_2D' and 'tth_integration_2D'. all 3!
    these dataset have to allready exist with the right shape and dtype and no compression!
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)
    # import inspect

    # linno = inspect.currentframe().f_lineno
    # print('DEBUG:\nin ' + __file__ + '\nline '+str(linno))
    # print(unpickled_args)

    source_fname = unpickled_args[0]
    source_datasetpath = unpickled_args[1]
    source_index_list = unpickled_args[2]
    edf_savepath = unpickled_args[3]
    edf_prefix = unpickled_args[4]
    target_index_list = unpickled_args[5]
    troi = unpickled_args[6]
    verbose = unpickled_args[7]

    if len(unpickled_args)>8:
        timer_fname = unpickled_args[8]
    else:
        timer_fname = None

        
    if verbose:
        print('='*25)
        print('process {} is doing'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)
    
    
    with h5py.File(source_fname) as h5_file:

        frameshape = h5_file[source_datasetpath][source_index_list[0]].shape

        if type(troi) == type(None):
            slices = troi_to_slice(((0,0),(frameshape[0],frameshape[1])))
        else:
            slices = troi_to_slice(troi)
            
        for target_index, source_index in zip(target_index_list,source_index_list):
            edf_fname = edf_savepath + os.path.sep + edf_prefix + '{:06d}.edf'
            source_data = h5_file[source_datasetpath][source_index][slices[0],slices[1]]

            save_edf.save_edf(source_data, edf_fname)
            
        if verbose:
            print('process {} is saving frame {} as edf'.format(os.getpid(),target_index))

            # h5_file.flush()

            
    
    if verbose:
        print('process {} is done'.format(os.getpid()))
        print('='*25)

    
    if timer_fname != None:
        os.remove(timer_fname)


if __name__=='__main__':
    ''' 
    This is used by the local function write_data_to_edf_employer(pickledargs_fname),
    DO NOT CHANGE
    '''
    if len(sys.argv)!=2:
        print('usage : python integrate_troi_worker <pickled_instruction_list_fname>')
    pickledargs_fname = sys.argv[1]
    write_data_to_edf_worker(pickledargs_fname)
                        
