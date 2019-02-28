from __future__ import print_function

import numpy as np
import h5py
import os, sys
from scipy.interpolate import interp1d
from xrayutilities import FuzzyGridder3D
from scipy.ndimage.measurements import center_of_mass as com


sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
import pythonmisc.my_xrayutilities as my_xu
from simplecalc.slicing import rebin, troi_to_slice

from simplecalc.calc import calc_sd

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
    troi_dict = unpickled_args[1]
    target_fname = unpickled_args[2]
    i_list = [int(x) for x in list(unpickled_args[3])]
    j_list = [int(x) for x in list(unpickled_args[4])]
    nx,ny,nz = [int(x) for x in unpickled_args[5]]
    kappa = float(unpickled_args[6])
    interp_factor = float(unpickled_args[7])
    verbose = unpickled_args[8]


    troiname = troi_dict['troiname']
    troi = troi_dict['troi']
    
    for key, value in troi_dict.items():
        print(key+' '+ str(value) + str(type(value)))

    if verbose:
        print('='*25)
        print('process {} is regrouping'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)
        print('='*25)
    
    with h5py.File(source_fname,'r') as source_file:
        Theta = 0
        troi_g = source_file['merged_data/diffraction/'+troiname]
        
        phi_list = []
        data_list = []
        for key, scan in  troi_g['single_scans'].items():
            phi_list.append(float(scan['phi'].value))
            data_list.append(scan['data'])
        # sort by phi value
        sort_list = zip(phi_list, data_list)
        sort_list.sort()
        phi_list = [x for x,y in sort_list]
        data_list = [y for x,y in sort_list]

        # this is dodgy: interpolation frames between sample position (phi)
        fine_phi_list = list(np.linspace(phi_list[0],phi_list[-1],(len(phi_list)-1)*interp_factor +1))
    
        # find Q regrouping, memory heavy!
        xu_exp = my_xu.get_id13_experiment(troi, troi_dict)
        qx, qy, qz = xu_exp.Ang2Q.area(Theta,kappa,fine_phi_list)

        # calculate alternate cordinate transformation to q, inplane angle and outofplane angle
        q_3d = (qx**2+qy**2+qz**2)**0.5
        in_plane = np.arctan2(qy,qx)*180/np.pi
        out_plane = np.arccos(qz/q_3d)*180/np.pi
                
        qxyz_gridder = FuzzyGridder3D(nx,ny,nz)
        qio_gridder = FuzzyGridder3D(nx,ny,nz)
        
        with h5py.File(target_fname,'w') as target_file:
            data_supergroup = target_file.create_group('entry/data')
            axes_group_exists = False
            
            for i, j in zip(i_list, j_list):
                data_group = data_supergroup.create_group('qxyz_{}_{:06d}_{:06d}'.format(troiname,i,j))
                
                if verbose:
                    print('getting data from {}'.format(source_fname))
                # read data and bin:
                raw_ds_dtype=np.uint64
                raw_data = np.asarray([x[i,j] for x in data_list],dtype = np.uint64)
                dtype=raw_ds_dtype
                                      
                if verbose:
                    print('interpolating {}'.format(source_fname))
                    print(len(phi_list),raw_data.shape, dtype)

                f = interp1d(phi_list, raw_data, axis=0, assume_sorted=True)
                interp_data = f(fine_phi_list)

                if verbose:
                    print('regridding {}\n saving {}'.format(source_fname, target_fname))
                    print('realspace grid: ',qx.shape, qx.dtype)
                    print('data: ', interp_data.shape, interp_data.dtype)

                qxyz_gridder(qx,qy,qz,interp_data)

                # do q_profile, this may be slow:
                dummy_data = np.zeros_like(interp_data)
                q_profile = np.zeros(shape=[len(fine_phi_list),nx],dtype=np.int64)
                ia_profile = np.zeros(shape=[len(fine_phi_list),ny],dtype=np.int64)
                oa_profile = np.zeros(shape=[len(fine_phi_list),nz],dtype=np.int64)
                                                
                for i in range(len(fine_phi_list)):
                    dummy_data *= 0
                    dummy_data[i] = interp_data[i]
                    qio_gridder(q_3d,in_plane,out_plane,dummy_data)
                    grid_data =  qio_gridder.data
                    sum_m1 = grid_data.sum(axis=-1)
                    q_profile[i] = sum_m1.sum(axis=1)
                    ia_profile[i] = sum_m1.sum(axis=0)
                    oa_profile[i] = grid_data.sum(axis=0).sum(axis=0)
                
                qio_gridder(q_3d,in_plane,out_plane,interp_data)

                qxyz_data = np.asarray(qxyz_gridder.data,dtype=dtype)
                qxyz_ds = data_group.create_dataset(name='qxyz_data', data = qxyz_data, compression='lzf')

                qio_data = np.asarray(qio_gridder.data,dtype=dtype)
                qio_ds = data_group.create_dataset(name='qio_data', data = qio_data, compression='lzf')
                data_group.create_dataset(name='ia_profile', data = ia_profile, compression='lzf')
                data_group.create_dataset(name='oa_profile', data = oa_profile, compression='lzf')
                data_group.create_dataset(name='q_profile', data = q_profile, compression='lzf')
                
                if not axes_group_exists:
                    qx_ax = qxyz_gridder.xaxis
                    qy_ax = qxyz_gridder.yaxis
                    qz_ax = qxyz_gridder.zaxis

                    q_ax = qio_gridder.xaxis
                    ia_ax = qio_gridder.yaxis
                    oa_ax = qio_gridder.zaxis
                                                            
                    q_axes = [qx_ax,qy_ax,qz_ax]
                    qio_axes = [q_ax,ia_ax,oa_ax]
                                                            
                    axes_group = target_file.create_group('entry/axes')
                    axes_group.create_dataset(name='qx', data = qx_ax, compression='lzf')
                    axes_group.create_dataset(name='qy', data = qy_ax, compression='lzf')
                    axes_group.create_dataset(name='qz', data = qz_ax, compression='lzf')
                    axes_group.create_dataset(name='q', data = q_ax, compression='lzf')
                    axes_group.create_dataset(name='ia', data = ia_ax, compression='lzf')
                    axes_group.create_dataset(name='oa', data = oa_ax, compression='lzf')
                    axes_group.create_dataset(name='Theta', data = Theta)
                    axes_group.create_dataset(name='kappa', data = kappa)
                    axes_group.create_dataset(name='phi', data = np.asarray(fine_phi_list))
                    axes_group.create_dataset(name='troi',data=troi)
                    axes_group_exists = True

                # calc realspace point values
                
                data_sum = qxyz_data.sum()
                data_max = qxyz_data.max()
                i_COM = com(qxyz_data)
                qx_com, qy_com, qz_com = q_com = np.asarray([np.interp(x,range(len(q_axes[l])),q_axes[l]) for l,x in enumerate(i_COM)])
                q_qxyz = (q_com**2).sum()**0.5
                sx, sy, sz = sigma = calc_sd(qxyz_data, data_sum, q_com, q_axes)
                s = (sigma**2).sum()**0.5

                qio_i_COM = com(qio_data)
                q_qio = np.interp(qio_i_COM[0],range(len(q_ax)),q_ax)
                ia_qio = np.interp(qio_i_COM[1],range(len(ia_ax)),ia_ax)
                oa_qio = np.interp(qio_i_COM[2],range(len(oa_ax)),oa_ax)

                
                # angles
                # oa is form the N pole down
                oa = np.arccos(abs(qz_com)/q_qxyz)
                pitch = np.arccos(qx_com/q_qxyz)
                roll = np.arccos(qy_com/q_qxyz)
                # get the 360deg version of ia
                ia = np.arctan2(qy_com,qx_com)
                
                # save realspace point values
                data_group.create_dataset(name='max',data=data_max)
                data_group.create_dataset(name='sum',data=data_sum)


                data_group.create_dataset(name='qx',data=qx_com)
                data_group.create_dataset(name='qy',data=qy_com)
                data_group.create_dataset(name='qz',data=qz_com)
                data_group.create_dataset(name='q_qxyz',data=q_qxyz)
                data_group.create_dataset(name='q_qio',data=q_qio)

                data_group.create_dataset(name='sx',data=sx)
                data_group.create_dataset(name='sy',data=sy)
                data_group.create_dataset(name='sz',data=sz)
                data_group.create_dataset(name='s',data=s)

                data_group.create_dataset(name='oa',data=oa)
                data_group.create_dataset(name='ia',data=ia)
                data_group.create_dataset(name='oa_qio',data=oa_qio)
                data_group.create_dataset(name='ia_qio',data=ia_qio)
                data_group.create_dataset(name='roll',data=roll)
                data_group.create_dataset(name='pitch',data=pitch)
                # target_file.flush

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
                        
