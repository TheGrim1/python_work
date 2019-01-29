from __future__ import print_function

import numpy as np
import h5py
import os, sys
from scipy.interpolate import interp1d
from xrayutilities import FuzzyGridder3D


from scipy.ndimage.measurements import center_of_mass as com   

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

import pythonmisc.my_xrayutilities as my_xu
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import rebin, troi_to_slice
import fileIO.spec.spec_tools as st

from simplecalc.calc import calc_sd

def id01qxzy_regroup_employer(pickledargs_fname):
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

def id01qxyz_regroup_worker(pickledargs_fname):
    '''
    copies troi into target_fname[target_datasetpath][target_index] from source_name[source_datasetpath][source_index][troi]
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)
    # import inspect

    # linno = inspect.currentframe().f_lineno
    # print('DEBUG:\nin ' + __file__ + '\nline '+str(linno))
    # print(unpickled_args)

    
    source_fname = unpickled_args[0]
    target_fname_list = unpickled_args[1]

    i_list = [int(x) for x in list(unpickled_args[2])]
    j_list = [int(x) for x in list(unpickled_args[3])]
    nx,ny,nz = [int(x) for x in unpickled_args[4]]

    map_shape = [int(x) for x in list(unpickled_args[5])]
    par_dict = unpickled_args[6]
    interp_factor = int(unpickled_args[7])
    verbose = unpickled_args[8]

    troi = par_dict['troi']
    bin_size = par_dict['bin_size']
    
    if verbose:
        print('='*25)
        print('process {} is regrouping'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)
        print('='*25)


    with h5py.File(source_fname,'r') as source_h5:

        eta_list = []
        delta_list = []
        data_list = []
        for key,scan in source_h5.items():
            scan_header = scan['instrument/specfile/scan_header'].value
            exp_angle_dict = st.get_ID01_rotations_from_scan_header(scan_header)
            eta_list.append(exp_angle_dict['eta'])
            delta_list.append(exp_angle_dict['del'])
            data_list.append(scan['instrument/detector/data'])

        # sort by eta value
        sort_list = zip(eta_list, data_list, delta_list)
        sort_list.sort()
        eta_list = [x for x,y,z in sort_list]
        data_path_list = [y for x,y,z in sort_list]
        delta_list = [z for x,y,z in sort_list]

        # this is dodgy: interpolation frames between sample position (eta) AND detector angles (delta):
        fine_eta_list = list(np.linspace(eta_list[0],eta_list[-1],(len(eta_list)-1)*interp_factor +1))
        fine_delta_list = list(np.linspace(delta_list[0],delta_list[-1],(len(delta_list)-1)*interp_factor +1))

        # maybe it will make sense to find a real crystal orientation
        phi   = exp_angle_dict['phi']

        # find Q regrouping, memory heavy!
        xu_exp = my_xu.get_id01_experiment(par_dict)
        qx, qy, qz = xu_exp.Ang2Q.area(fine_eta_list,phi,fine_delta_list)
        gridder = FuzzyGridder3D(nx,ny,nz)

        with h5py.File(target_fname_list[0],'w') as target_file:
            data_supergroup = target_file.create_group('entry/data')
            axes_group_exists = False
            for i, j, ij_fname in zip(i_list, j_list, target_fname_list):

                data_group = data_supergroup.create_group(os.path.splitext(os.path.basename(ij_fname))[0])
                
                if verbose:
                    print('getting data from {}'.format(source_fname))
                index = i*map_shape[1]+j

                # read data and bin:
                raw_data = np.asarray([rebin(x[index][troi_to_slice(troi)],[bin_size]*2) for x in data_list], dtype=np.float64)
                print('raw_data.shape = ',raw_data.shape)
                print('qx.shape =  ', qx.shape)
                dtype=raw_data.dtype
                if verbose:
                    print('interpolating factor {}'.format(interp_factor))
                    print(len(eta_list),raw_data.shape, dtype)

                f = interp1d(eta_list, raw_data, axis=0, assume_sorted=True)
                interp_data = f(fine_eta_list)

                if verbose:
                    print('regridding {}\n saving {}'.format(source_fname, ij_fname))
                    print('realspace grid: ', qx.shape, qx.dtype)
                    print('data: ', interp_data.shape, interp_data.dtype)
                    
                gridder(qx,qy,qz,interp_data)
                data = np.asarray(gridder.data,dtype=dtype)
                ds = data_group.create_dataset(name='data', data = data, compression='lzf')
                
                if not axes_group_exists:
                    qx_ax = gridder.xaxis
                    qy_ax = gridder.yaxis
                    qz_ax = gridder.zaxis
                    q_axes = [qx_ax,qy_ax,qz_ax]
                                                            
                    axes_group = target_file.create_group('entry/axes')
                    axes_group.create_dataset(name='qx', data = qx_ax, compression='lzf')
                    axes_group.create_dataset(name='qy', data = qy_ax, compression='lzf')
                    axes_group.create_dataset(name='qz', data = qz_ax, compression='lzf')
                    axes_group.create_dataset(name='eta', data = np.asarray(fine_eta_list), compression='lzf')
                    axes_group.create_dataset(name='delta', data = np.asarray(fine_delta_list), compression='lzf')
                    axes_group_exists = True

                # calc realspace point values
                data_sum = data.sum()
                data_max = data.max()
                i_COM = com(data)
                qx_com, qy_com, qz_com = q_com = np.asarray([np.interp(x,range(len(q_axes[i])),q_axes[i]) for i,x in enumerate(i_COM)])
                q = (q_com**2).sum()**0.5
                sx, sy, sz = sigma = calc_sd(data, data_sum, q_com, q_axes)
                s = (sigma**2).sum()**0.5
                
                # angles
                theta = np.arccos(abs(qz_com)/q)
                pitch = np.arccos(qx_com/q)
                roll = np.arccos(qy_com/q)
                phi = np.arctan(qx_com/qy_com)

                # save realspace point values
                data_group.create_dataset(name='max',data=data_max)
                data_group.create_dataset(name='sum',data=data_sum)

                data_group.create_dataset(name='qx',data=qx_com)
                data_group.create_dataset(name='qy',data=qy_com)
                data_group.create_dataset(name='qz',data=qz_com)
                data_group.create_dataset(name='q',data=q)

                data_group.create_dataset(name='sx',data=sx)
                data_group.create_dataset(name='sy',data=sy)
                data_group.create_dataset(name='sz',data=sz)
                data_group.create_dataset(name='s',data=s)

                data_group.create_dataset(name='theta',data=theta)
                data_group.create_dataset(name='phi',data=phi)
                data_group.create_dataset(name='roll',data=roll)
                data_group.create_dataset(name='pitch',data=pitch)

                
            target_file.flush()
                
                
                
            target_file.flush()
            
                    

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
    id01qxyz_regroup_worker(pickledargs_fname)
                        
