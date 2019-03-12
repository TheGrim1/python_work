from __future__ import print_function

import numpy as np
import h5py
import os, sys

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
import simplecalc.fitting3d as fit3d
import simplecalc.fitting2d as fit2d
import simplecalc.slicing as sl

def fit_employer(pickledargs_fname):
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

def fit_worker(pickledargs_fname):
    '''
    do 2d and 3d gaussian fit to the various datasets in a qmerged file
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)
    # import inspect

    # linno = inspect.currentframe().f_lineno
    # print('DEBUG:\nin ' + __file__ + '\nline '+str(linno))
    # print(unpickled_args)

    
    source_fname = unpickled_args[0]
    target_fname = unpickled_args[1]
    troiname = unpickled_args[2]
    i_list = [int(x) for x in list(unpickled_args[3])]
    j_list = [int(x) for x in list(unpickled_args[4])]
    mask = np.asarray(unpickled_args[5])
    binning = np.asarray(unpickled_args[6])
    verbose = unpickled_args[7]
    pid = os.getpid()
    if verbose:
        print('='*25)
        print('process {} is regrouping'.format(pid))
        for arg in unpickled_args:
            print(arg)
        print('='*25)
    
    with h5py.File(source_fname,'r') as source_file:

        troi_g = source_file['diffraction/'+troiname]
        axes_g = source_file['axes/'+troiname]
        # coordinates are sorted array[y,x,z]! so data with [qx,qy] has xx = qy and yy = qx
        qy, qx, qz = np.meshgrid(axes_g['qy'],axes_g['qx'],axes_g['qz'])
        ia, q, oa = np.meshgrid(axes_g['ia'],axes_g['q'],axes_g['oa'])
        q_phi, q_fine_phi = np.meshgrid(axes_g['q'],source_file['axes/fine_phi'])
        ia_phi, ia_fine_phi = np.meshgrid(axes_g['ia'],source_file['axes/fine_phi'])
        oa_phi, oa_fine_phi = np.meshgrid(axes_g['oa'],source_file['axes/fine_phi'])
        
        with h5py.File(target_fname,'w') as target_file:
            data_supergroup = target_file.create_group('entry/data')

            fit3d_result = np.empty(shape=(2,10),dtype=np.float64)
            fit2d_result = np.empty(shape=(2,6),dtype=np.float64)
            residual3d = np.zeros_like(qx)
            
            for i, j in zip(i_list, j_list):
                data_group = data_supergroup.create_group('fit_{}_{:06d}_{:06d}'.format(troiname,i,j))
                # 3d data
                # again coordinates are sorted array[y,x,z]! so data with [qx,qy] has xx = qy and yy = qx
                # see np.meshgrid
                # fit3d/fit2d result parameters are soreted as [y,x,(z)] also!
                data3d_dict = {'Qxyz/data_all':{'xx':qy,
                                                'yy':qx,
                                                'zz':qz,
                                                'parameters':['qx','qy','qz','sx','sy','sz','sxy','sxz','syz','A']},
                               'Qio/data_all':{'xx':ia,
                                               'yy':q,
                                               'zz':oa,
                                               'parameters':['q','ia','oa','sq','sia','soa','sqia','sqoa','siaoa','A']}}
                
                for dataname, axes in data3d_dict.items():

                    ds_g = data_group.create_group(dataname)
                    if mask[i,j]:
                        if verbose > 4:
                            print('i,j,bin')
                            print(i,j,binning)
                        if binning:
                            bin_sl = sl.troi_to_slice(sl.make_troi([i,j],binning))
                            data = np.median(np.median(np.asarray(troi_g[dataname][bin_sl]),axis=0),axis=0)
                        else:
                            data = np.asarray(troi_g[dataname][i,j])
                            
                        if verbose >2:
                            print('pid: {} fitting to {:04d} {:04d}'.format(pid, i,j))
                            print('dataname {}'.format(dataname))
                            print('data.shape', data.shape)
                            print('xx.shape', axes['xx'].shape)
                                  
                        fit3d_result, residual3d = fit3d.do_iterative_two_gauss3d_fit(data=data,xx=axes['xx'],yy=axes['yy'],zz=axes['zz'],force_positive=True,diff_threshold=0.002, max_iteration=4, return_residual=True, verbose=verbose)
                    else:
                        fit3d_result.fill(np.nan)
                        residual3d.fill(np.nan)
                    ds = ds_g.create_dataset('fit3d_result',data=fit3d_result)
                    ds = ds_g.create_dataset('residual',data=residual3d)

                    if sys.version_info < (3,):
                        string_dtype = h5py.special_dtype(vlen=unicode)
                    else:
                        string_dtype = h5py.special_dtype(vlen=str) 
                    ds_g.attrs['parameters'] = np.array(axes['parameters'], dtype=string_dtype)
                
                    
                # 2d data
                data2d_dict = {'Qio/ia_profile':{'xx':ia_phi,
                                                 'yy':ia_fine_phi,
                                                 'parameters':['phi','ia','sphi','sia','rho','A']},
                               'Qio/q_profile':{'xx':q_phi,
                                                'yy':q_fine_phi,
                                                'parameters':['phi','q','sphi','sq','rho','A']},
                               'Qio/oa_profile':{'xx':oa_phi,
                                                 'yy':oa_fine_phi,
                                                 'parameters':['phi','oa','sphi','soa','rho','A']}}


                for dataname, axes in data2d_dict.items():
                    residual2d = np.zeros_like(axes['xx'])
                    ds_g = data_group.create_group(dataname)
                    if mask[i,j]:
                        if binning:
                            bin_sl = sl.troi_to_slice(sl.make_troi([i,j],binning))
                            data = np.median(np.median(np.asarray(troi_g[dataname][bin_sl]),axis=0),axis=0)
                        else:
                            data = np.asarray(troi_g[dataname][i,j])

                        
                        data = np.asarray(troi_g[dataname][i,j])
                        if verbose >2:
                            print('dataname {}'.format(dataname))
                            print('data.shape', data.shape)
                            print('xx.shape', axes['xx'].shape)
                                  
                        fit2d_result, residual2d = fit2d.do_iterative_two_gauss2d_fit(data=data,xx=axes['xx'],yy=axes['yy'],force_positive=True,diff_threshold=0.002, max_iteration=4, return_residual=True, verbose=verbose)
                    else:
                        fit2d_result.fill(np.nan)
                        residual2d.fill(np.nan)
                        
                    ds = ds_g.create_dataset('fit2d_result',data=fit2d_result)
                    ds = ds_g.create_dataset('residual',data=residual2d)           
                    if sys.version_info < (3,):
                        string_dtype = h5py.special_dtype(vlen=unicode)
                    else:
                        string_dtype = h5py.special_dtype(vlen=str) 
                    ds_g.attrs['parameters'] = np.array(axes['parameters'], dtype=string_dtype)
                

            target_file.flush()

    if verbose:
        print('process {} is done'.format(pid))
        print('='*25)


        
if __name__=='__main__':
    ''' 
    This is used by the local function fit_employer(pickledargs_fname),
    DO NOT CHANGE
    '''
    if len(sys.argv)!=2:
        print('usage : python fit_worker <pickled_instruction_list_fname>')
    pickledargs_fname = sys.argv[1]
    fit_worker(pickledargs_fname)
                        
