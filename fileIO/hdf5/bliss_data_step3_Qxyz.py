import h5py
import sys,os
import time
import glob
from shutil import rmtree
import numpy as np

from multiprocessing import Pool


sys.path.append('/data/id13/inhouse2/AJ/skript')
import fileIO.spec.spec_tools as st
from pythonmisc.worker_suicide import worker_init
from fileIO.hdf5.workers import bliss_qxyz_regroup_worker as qrw
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import rebin, troi_to_slice
from fileIO.pyFAI.poni_for_troi import poni_for_troi


def parse_ij(fname):
    # tpl = qxyz_redrid_{:06d}_{:06d}.h5
    i,j=[int(x) for x in os.path.splitext(fname)[0].split('_')[-2::]]
    troiname = os.path.splitext(fname)[0].split('_')[-3]
    return i,j

def do_regrouping(merged_fname, poni_fname, Q_dim, interp_factor, prefix='', verbose = True):
    Qmerged_fname = os.path.dirname(merged_fname) + os.path.sep + prefix + 'qxyz_'+os.path.basename(merged_fname)
    if os.path.exists(Qmerged_fname):
        os.remove(Qmerged_fname)
    starttime = time.time()
    total_datalength = 0    ### get the need info:
    no_processes = 30
    
    with h5py.File(merged_fname,'r') as source_h5:
        
        dest_dir = os.path.dirname(merged_fname)+os.path.sep+prefix + os.path.splitext(os.path.basename(merged_fname))[0] + '/'
        if os.path.exists(dest_dir):
            print('deleting old data in {}'.format(dest_dir))
            rmtree(dest_dir)
        os.mkdir(dest_dir)

        map_shape = tuple(source_h5['merged_data/fluorescence/fluo_aligned/XRF'].shape[1:])
        Qdata_shape = tuple(list(map_shape) + list(Q_dim))
        diff_g = source_h5['merged_data/diffraction']
        kappa = source_h5['merged_data/axes/kappa'].value
        # setup todolist for parrallel processes per map point


        todo_list = []
        troiname_list = []
        
        for troiname in diff_g.keys():
            troiname_list.append(troiname)
            raw_ds_dtype = np.uint64
            dest_fname_tpl = dest_dir + 'qxyz_{}_{}_{}.h5'.format(troiname,'{:06d}','{:06d}')
            troi = np.asarray(diff_g[troiname]['troi'])
            troi_dict = poni_for_troi(poni_fname, troi=troi, troiname=troiname)[0]
            troi_dict['troiname']=troiname
            troi_dict['troi']=troi
            
            for i in range(map_shape[0]):
                i_list = [i] * map_shape[1]
                j_list = range(map_shape[1])
                # j_list = range(10)
                total_datalength += map_shape[1]
                todo=[merged_fname,
                      troi_dict,
                      dest_fname_tpl.format(i,j_list[0]),
                      i_list,
                      j_list,
                      Q_dim,
                      kappa,
                      interp_factor,
                      False]
                todo_list.append(todo)

    print('setup parallel proccesses to write to {}'.format(dest_dir))
    instruction_list = []
    for i,todo in enumerate(todo_list):
        instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=False, counter=i)
        instruction_list.append(instruction_fname)
    if no_processes==1:
        print(instruction_list)
        for i, instruction in enumerate(instruction_list):
            print('running in single process, loop {}'.format(i))
            qrw.qxyz_regroup_worker(instruction)

        ## non parrallel version for one dataset and timing:
        #fdw.fit_data_worker(instruction_list[0])
    else:
        pool = Pool(no_processes, worker_init(os.getpid()))
        pool.map_async(qrw.qxzy_regroup_employer,instruction_list)
        pool.close()
        pool.join()


    endreadtime = time.time()
    total_read_time = (endreadtime - starttime)
    print('='*25)
    print('\ntime taken for regrouping of {} datasets = {}'.format(total_datalength, total_read_time))
    print(' = {} Hz\n'.format(total_datalength/total_read_time))
    print('='*25) 
             
    result_fname_list = glob.glob(dest_dir+os.path.sep+'*.h5')
     

    with h5py.File(Qmerged_fname,'w') as dest_h5:
        print('writing to merged file {}'.format(Qmerged_fname))

        with h5py.File(merged_fname,'r') as merged_h5:

            merged_h5.copy('merged_data/axes', dest_h5,'axes')
            merged_h5.copy('merged_data/alignment', dest_h5,'alignment')
            merged_h5.copy('merged_data/fluorescence', dest_h5,'fluorescence')

            
            axes_group = dest_h5['axes']
            
        for troiname in troiname_list:
            troi_fname_list = [x for x in result_fname_list if x.find('_'+troiname+'_')>0]

            troi_g = dest_h5.create_group('diffraction/'+troiname)
            troi_qaxes_group = axes_group.create_group(troiname) 
            q_group = troi_g.create_group('Qxyz')
            qio_group = troi_g.create_group('Qio')
            
            s_group = troi_g.create_group('Sxys')
            sum_group = troi_g.create_group('sum')
            max_group = troi_g.create_group('max')
            
            # get dtype
            with h5py.File(troi_fname_list[0],'r') as source_h5:
                raw_ds_dtype = np.uint64
                axes_dict = {'qx':'',
                             'qy':'',
                             'qz':'',
                             'q':'',
                             'ia':'',
                             'oa':''}

                source_axes = source_h5['entry/axes']
                for axis, ax_data in source_axes.items():                    
                    if axis in axes_dict.keys():
                        axes_dict[axis]=np.asarray(ax_data)
                        
                        if not axis in troi_qaxes_group.keys():
                            ax_ds = troi_qaxes_group.create_dataset(axis,data=ax_data)
                            ax_ds.attrs['long_name'] = axis + ' in inv. nm'

                if not 'fine_phi' in axes_group.keys():
                    phi_ds = axes_group.create_dataset('fine_phi', data=np.asarray(source_axes['phi']))
                    fine_phi_points = phi_ds.shape[0]
                
                for group in [sum_group, max_group]:
                    group['qx'] = troi_qaxes_group['qx']
                    group['qy'] = troi_qaxes_group['qy']
                    group['qz'] = troi_qaxes_group['qz']
                    group.attrs['signal'] = u'q_space'
                    group.attrs['NXClass'] = 'NXData'

                    if sys.version_info < (3,):
                        string_dtype = h5py.special_dtype(vlen=unicode)
                    else:
                        string_dtype = h5py.special_dtype(vlen=str)
                        group.attrs['axes'] = numpy.array(['qx', 'qy', 'qz'], dtype=string_dtype)
                        
                    
                q_axes = [axes_dict['qx'], axes_dict['qy'], axes_dict['qz']]


                
            profile_shape = tuple(list(map_shape) + [fine_phi_points, Qdata_shape[3]])
                                      
            pointwise_group = q_group.create_group('single_files')
            chunks = (1,1,len(q_axes[0]),len(q_axes[1]),len(q_axes[2]))
            q_all_ds = q_group.create_dataset(name='data_all', shape=Qdata_shape, dtype=np.uint64, compression='lzf', chunks=chunks)
            qio_all_ds = qio_group.create_dataset(name='data_all', shape=Qdata_shape, dtype=np.uint64, compression='lzf', chunks=chunks)

            q_profile_ds = qio_group.create_dataset(name='q_profile', shape=profile_shape, dtype=np.uint64, compression='lzf', chunks=(1,1,fine_phi_points, Qdata_shape[3]))
            ia_profile_ds = qio_group.create_dataset(name='ia_profile', shape=profile_shape, dtype=np.uint64, compression='lzf', chunks=(1,1,fine_phi_points, Qdata_shape[3]))
            oa_profile_ds = qio_group.create_dataset(name='oa_profile', shape=profile_shape, dtype=np.uint64, compression='lzf', chunks=(1,1,fine_phi_points, Qdata_shape[3]))

            qx_ar = np.zeros(shape=map_shape, dtype = np.float64)
            qy_ar = np.zeros(shape=map_shape, dtype = np.float64)
            qz_ar = np.zeros(shape=map_shape, dtype = np.float64)
            q_ar = np.zeros(shape=map_shape, dtype = np.float64)
            q_qio_ar = np.zeros(shape=map_shape, dtype = np.float64)

            sx_ar = np.zeros(shape=map_shape, dtype = np.float64)
            sy_ar = np.zeros(shape=map_shape, dtype = np.float64)
            sz_ar = np.zeros(shape=map_shape, dtype = np.float64)
            s_ar = np.zeros(shape=map_shape, dtype = np.float64)

            oa_ar = np.zeros(shape=map_shape, dtype = np.float64)
            ia_ar = np.zeros(shape=map_shape, dtype = np.float64)
            oa_qio_ar = np.zeros(shape=map_shape, dtype = np.float64)
            ia_qio_ar = np.zeros(shape=map_shape, dtype = np.float64)
            roll_ar = np.zeros(shape=map_shape, dtype = np.float64)
            pitch_ar = np.zeros(shape=map_shape, dtype = np.float64)


            curr_qmax = np.zeros(shape=Qdata_shape[2:], dtype=raw_ds_dtype)
            curr_qsum = np.zeros(shape=Qdata_shape[2:], dtype=raw_ds_dtype)

            curr_rsum = np.zeros(shape=map_shape, dtype=raw_ds_dtype)
            curr_rmax = np.zeros(shape=map_shape, dtype=raw_ds_dtype)

            for fname in troi_fname_list:
                print('collecting {}'.format(fname))
                f_i,f_j = parse_ij(fname)
                file_group = pointwise_group.create_group(os.path.basename(fname))

                with h5py.File(fname,'r') as source_h5:

                    for i_j, dg in source_h5['entry/data'].items():

                        print('collecting group {}'.format(i_j))
                        r_i,r_j = parse_ij(i_j)
                        frame_no = r_i*map_shape[1]+r_j

                        file_group.create_dataset('{}'.format(frame_no),data='point_{:06d}_{:06d}'.format(r_i,r_j))
                        # raw data:
                        data_ij = np.asarray(dg['qxyz_data'],dtype=raw_ds_dtype)
                        q_all_ds[r_i,r_j] = data_ij
                        qio_data_ij = np.asarray(dg['qio_data'],dtype=raw_ds_dtype)
                        qio_all_ds[r_i,r_j] = data_ij

                        q_profile_data_ij = np.asarray(dg['q_profile'],dtype=raw_ds_dtype)
                        q_profile_ds[r_i,r_j] = q_profile_data_ij
                        ia_profile_data_ij = np.asarray(dg['ia_profile'],dtype=raw_ds_dtype)
                        ia_profile_ds[r_i,r_j] = ia_profile_data_ij
                        oa_profile_data_ij = np.asarray(dg['oa_profile'],dtype=raw_ds_dtype)
                        oa_profile_ds[r_i,r_j] = oa_profile_data_ij

                        curr_qmax = np.where(data_ij>curr_qmax,data_ij,curr_qmax)
                        curr_qsum += data_ij
                        curr_rmax[r_i,r_j] = np.float(dg['max'].value)
                        data_ij_sum = np.float(dg['sum'].value)
                        curr_rsum[r_i,r_j] = data_ij_sum

                        qx_ar[r_i,r_j] = np.float(dg['qx'].value)
                        qy_ar[r_i,r_j] = np.float(dg['qy'].value)
                        qz_ar[r_i,r_j] = np.float(dg['qz'].value)
                        q_ar[r_i,r_j] = np.float(dg['q_qxyz'].value)
                        q_qio_ar[r_i,r_j] = np.float(dg['q_qio'].value)
                                                

                        sx_ar[r_i,r_j] = np.float(dg['sx'].value)
                        sy_ar[r_i,r_j] = np.float(dg['sy'].value)
                        sz_ar[r_i,r_j] = np.float(dg['sz'].value)
                        s_ar[r_i,r_j] = np.float(dg['s'].value)

                        oa_ar[r_i,r_j] = np.float(dg['oa'].value)
                        ia_ar[r_i,r_j] = np.float(dg['ia'].value)
                        oa_qio_ar[r_i,r_j] = np.float(dg['oa_qio'].value)
                        ia_qio_ar[r_i,r_j] = np.float(dg['ia_qio'].value)
                        roll_ar[r_i,r_j] = np.float(dg['pitch'].value)
                        pitch_ar[r_i,r_j] = np.float(dg['roll'].value)


            qx_ds = q_group.create_dataset(name='qx', data = qx_ar)
            qy_ds = q_group.create_dataset(name='qy', data = qy_ar)
            qz_ds = q_group.create_dataset(name='qz', data = qz_ar)
            q_ds = q_group.create_dataset(name='q', data = q_ar)
            q_ds = qio_group.create_dataset(name='q', data = q_qio_ar)

            sx_ds = s_group.create_dataset(name='sx', data = sx_ar)
            sy_ds = s_group.create_dataset(name='sy', data = sy_ar)
            sz_ds = s_group.create_dataset(name='sz', data = sz_ar)
            s_ds = s_group.create_dataset(name='s', data = s_ar)

            oa_ds = q_group.create_dataset(name='oa', data = oa_ar * 180./np.pi)
            ia_ds = q_group.create_dataset(name='ia', data = ia_ar * 180./np.pi)
            oa_qio_ds = qio_group.create_dataset(name='oa', data = oa_qio_ar)
            ia_qio_ds = qio_group.create_dataset(name='ia', data = ia_qio_ar)
            
            roll_ds = q_group.create_dataset(name='roll', data = roll_ar * 180./np.pi)
            pitch_ds = q_group.create_dataset(name='pitch', data = pitch_ar * 180./np.pi)

            max_ds = max_group.create_dataset(name='q_space', data=curr_qmax)
            sum_ds = sum_group.create_dataset(name='q_space', data=curr_qsum)
            max_ds = max_group.create_dataset(name='r_space', data=curr_rmax)
            sum_ds = sum_group.create_dataset(name='r_space', data=curr_rsum)


            # make dataset which masks accoring to XRF:
            q_masked_g = troi_g.create_group('Q_masked')
            nan_mask = np.asarray(dest_h5['fluorescence/fluo_aligned/mask'],dtype = np.float32)
            nan_mask[np.logical_not(nan_mask)] = np.nan

            qx_ds = q_masked_g.create_dataset(name='qx', data = qx_ar*nan_mask)
            qy_ds = q_masked_g.create_dataset(name='qy', data = qy_ar*nan_mask)
            qz_ds = q_masked_g.create_dataset(name='qz', data = qz_ar*nan_mask)
            q_ds = q_masked_g.create_dataset(name='q', data = q_qio_ar*nan_mask)

            sx_ds = q_masked_g.create_dataset(name='sx', data = sx_ar*nan_mask)
            sy_ds = q_masked_g.create_dataset(name='sy', data = sy_ar*nan_mask)
            sz_ds = q_masked_g.create_dataset(name='sz', data = sz_ar*nan_mask)
            s_ds = q_masked_g.create_dataset(name='s', data = s_ar*nan_mask)

            oa_ds = q_masked_g.create_dataset(name='oa', data = oa_ar * 180./np.pi*nan_mask)
            ia_ds = q_masked_g.create_dataset(name='ia', data = ia_ar * 180./np.pi*nan_mask)
            
            roll_ds = q_masked_g.create_dataset(name='roll', data = roll_ar * 180./np.pi*nan_mask)
            pitch_ds = q_masked_g.create_dataset(name='pitch', data = pitch_ar * 180./np.pi*nan_mask)

            q_masked_g.create_dataset(name='max_rspace', data=curr_rmax*nan_mask)
            q_masked_g.create_dataset(name='sum_rspace', data=curr_rsum*nan_mask*nan_mask)

            
    end_merge_time = time.time()
    total_merge_time = (end_merge_time - endreadtime)
    print('='*25)
    print('\ntime taken for merging of {} datasets = {}'.format(total_datalength, total_merge_time))
    print(' = {} Hz\n'.format(total_datalength/total_merge_time))
    print('='*25) 

    print('written to merged file {}'.format(Qmerged_fname))

    total_time = (end_merge_time - starttime)
    print('='*25)
    print('\ntotal time taken for {} datasets = {}'.format(total_datalength, total_time))
    print(' = {} Hz\n'.format(total_datalength/total_time))
    print('='*25) 
             

def main():

    # merged_fname = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/previews/alignment/kmap_rocking_merged.h5'
    # poni_fname = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/calib/calib1.poni'
    
    # # 
    # poni_fname = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/calib2/calib2.poni'
    # merged_fname = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/previews/day_two/kmap_and_cen_4b_merged.h5'

    poni_fname = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/calib2/calib2.poni'
    merged_fname = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/previews/day_two/kmap_and_cen_3b_merged.h5'

    
    interp_factor = 1
    Q_disc = 20
    Q_dim = [nQx, nQy, nQz] = [Q_disc]*3
    
    prefix = 'q{}_int{}_'.format(Q_disc,interp_factor)
    
    # check size of rebinned data:
    
    
    do_regrouping(merged_fname, poni_fname, Q_dim, interp_factor=interp_factor, prefix=prefix, verbose=True)

    
if __name__ == "__main__":
    main()
