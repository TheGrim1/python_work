from __future__ import print_function

import numpy as np
import h5py
import os, sys
from scipy.interpolate import interp1d
from xrayutilities import FuzzyGridder3D
import pythonmisc.my_xrayutilities as my_xu
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
    troi = np.asarray(unpickled_args[3])
    troi_poni = unpickled_args[4]
    [i,j] = [int(x) for x in list(unpickled_args[5])]
    nx,ny,nz = [int(x) for x in unpickled_args[6]]
    Theta_index_list = [int(x) for x in unpickled_args[7]]
    Theta_list = [float(x) for x in unpickled_args[8]]
    fine_Theta_list = [float(x) for x in unpickled_args[9]]
    [kappa,phi] = [float(x) for x in unpickled_args[10]]
    verbose = unpickled_args[11]

    for key, value in troi_poni.items():
        print(key+' '+ str(value) + str(type(value)))

    
    # find Q regrouping, memory heavy!
    xu_exp = my_xu.get_id13_experiment(troi, troi_poni)
    qx, qy, qz = xu_exp.Ang2Q.area(fine_Theta_list,kappa,phi)
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
                print('getting data from {}'.format(source_fname))
                print(Theta_index_list[0])
                print(Theta_index_list[-1])
            raw_data = np.asarray(source_file[source_ds_path][i,j][slice(Theta_index_list[0], Theta_index_list[-1]+1,1)])
            dtype=raw_data.dtype
            if verbose:
                print('interpolating {}'.format(source_fname))
                print(len(Theta_list),raw_data.shape, dtype)
                
            f = interp1d(Theta_list, raw_data, axis=0, assume_sorted=True)
            interp_data = f(fine_Theta_list)

            if verbose:
                print('regridding {}\n saving {}'.format(source_fname, target_fname))
                print('realspace grid: ', qx.shape, qx.dtype)
                print('data: ', interp_data.shape, interp_data.dtype)
                
            data_group = target_file.create_group('entry/data')
            gridder(qx,qy,qz,interp_data)
            data = np.asarray(gridder.data,dtype=dtype)
            ds = data_group.create_dataset(name='data', data = data, compression='lzf')
                
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
                        
