from __future__ import print_function

import numpy as np
import h5py
import os, sys
from scipy import interpolate.interp1d as interp1d
from xrayutils import FuzzyGridder3D 
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu


def qxzy_regroup_employer(pickledargs_fname):
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

def qxyz_regroup_worker(pickledargs_fname):
    '''
    copies troi into target_fname[target_datasetpath][target_index] from source_name[source_datasetpath][source_index][troi]
    these dataset have to allready exist with the right shape and dtype 
    no compression, if more than on onf these workers is working on one file! 
    Changes to unpickling here must be updated in h5_scan_nexusversion
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)
    # import inspect

    # linno = inspect.currentframe().f_lineno
    # print('DEBUG:\nin ' + __file__ + '\nline '+str(linno))
    # print(unpickled_args)

    
    source_fname = unpickled_args[0]
    source_ds_path = unpickled_args[1]
    target_fname = unpickled_args[2]
    [i,j] = unpickled_args[3]
    nx,ny,nz = unpickled_args[4]
    Theta_list = unpickled_args[5]
    fine_Theta_list = unpickled_args[6] 
    [qx, qy, qz] = unpickled_args[7]
    verbose = unpickled_args[7]

    gridder = FuzzyGridder3D(nx,ny,nz)
    
    if verbose:
        print('='*25)
        print('process {} is regrouping'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)
        print('='*25)
    
    with h5py.File(target_fname,'w') as target_file:
        with h5py.File(source_fname,'r') as source_file:
            if verbose:
                print('getting data')
            raw_data = np.asarray(target_fname[source_ds_path][i,j][slice(Theta_list[0], Theta_list[-1],1)])
            if verbose:
                print('interpolating')             
            f = interp1d(Theta_list, raw_data, order=1)
            interp_data = f(fine_Theta_list)

            if verbose:
                print('regridding, saving')
            data_group = target_file.create_group['entry/data']
            ds = data_group.create_dataset(name='data', data = np.asarray(gridder(qx,qy,qz,interp_data),dtype=raw_data.dtype, compression='lzf')
                
            # target_file.flush()
            
    if verbose:
        print('process {} is done'.format(os.getpid()))
        print('='*25)


        
if __name__=='__main__':
    ''' 
    This is used by the lacal function qxyz_regroup_employer(pickledargs_fname),
    DO NOT CHANGE
    '''
    if len(sys.argv)!=2:
        print('usage : python qxyz_regroup_worker <pickled_instruction_list_fname>')
    pickledargs_fname = sys.argv[1]
    qxyz_regroup_worker(pickledargs_fname)
                        
