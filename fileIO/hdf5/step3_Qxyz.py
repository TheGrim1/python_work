import h5py
import numpy as np
import sys, os
import datetime
import time
from multiprocessing import Pool
import glob

from shutil import rmtree

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from pythonmisc.worker_suicide import worker_init
from fileIO.hdf5.workers import qxyz_regroup_worker as qrw
import pythonmisc.pickle_utils as pu


def parse_ij(fname):
    # tpl = qxyz_redrid_{:06d}_{:06d}.h5
    i,j=[int(x) for x in os.path.splitext(fname)[0].split('_')[-2::]]
    return i,j

def parse_troiname(fname):
    # tpl = qxyz_redrid_{:06d}_{:06d}.h5
    troi=[str(x) for x in os.path.splitext(fname)[0].split('_')[-3]]
    return troi

def do_parallel_regrouping(merged_fname, Q_dim, interp_factor = 1, limit_Thetas=False, verbose = True):
    Qmerged_fname = os.path.dirname(merged_fname)+os.path.sep+'qxyz_'+os.path.basename(merged_fname)
    starttime = time.time()
    total_datalength = 0
    ### get the needed info:
    no_processes=1
    with h5py.File(merged_fname,'r') as source_h5:
        diff_group = source_h5['entry/merged_data/diffraction/']
        troi_list = diff_group.keys()
        Theta_list = np.asarray(source_h5['entry/merged_data/axes/Theta'])
        # maybe it will make sense to find a real crystal orientation
        kappa = 0
        phi   = 0
        dest_dir = os.path.dirname(merged_fname)+os.path.sep+'qxyz_regrouped/'
        if os.path.exists(dest_dir):
            print('deleting old data in {}'.format(dest_dir))
            rmtree(dest_dir)
        os.mkdir(dest_dir)
        todo_list=[]
        for troiname in troi_list:
            troi_path = 'entry/merged_data/diffraction/{}'.format(troiname)
            raw_ds_path = troi_path+'/raw_data/all/data'
            raw_ds = source_h5[raw_ds_path]
            raw_ds_dtype = raw_ds.dtype
            map_shape = list(raw_ds.shape[0:2])
            troi_poni = dict([[x,np.asarray(y)] for x,y in source_h5['entry/calibration/{}/'.format(troiname)].items()])
            ### change troi size to rebin, required for qxyz
            bin_size = source_h5['entry/merged_data/axes/{}/bin_size'.format(troiname)].value
            troi = np.asarray(source_h5['entry/merged_data/axes/{}/troi'.format(troiname)])
            troi = [troi[0],[int(troi[1][0]/bin_size),int(troi[1][1]/bin_size)]]
            
            Qdata_shape = map_shape + Q_dim

            # find which Thetas are good
            
            if limit_Thetas:
                good_indexes = []
                sum_ds = np.asarray(source_h5[troi_path+'/raw_data/all/data_sum'])
                Theta_sums = sum_ds.sum(axis=0).sum(axis=0)
                for i,Theta_sum in enumerate(Theta_sums):
                    if Theta_sum > 0.01 * max(Theta_sums):
                        good_indexes.append(i)
                ## add a frame before and after the < 0.01 threshold
                Theta_first = int(max(0,min(good_indexes)-1))
                Theta_last = int(min(len(Theta_list), max(good_indexes)+2))
                

            else:
                Theta_first = 0
                Theta_last = len(Theta_list)

            troi_Theta_index_list = range(Theta_first,Theta_last)
            troi_Theta_list = Theta_list[Theta_first:Theta_last]
            print('found data between Theta_indexes {} and {}'.format(Theta_first, Theta_last))
            print('found data in Theta ',troi_Theta_list)
            
            fine_Theta_list = list(np.linspace(troi_Theta_list[0],troi_Theta_list[-1],(len(troi_Theta_list)-1)*interp_factor +1))
            print('Theta list refined = ',fine_Theta_list)
            
            # setup todolist for parrallel processes per map point
            troi_dir = (dest_dir+os.path.sep+troiname+os.path.sep)
            os.mkdir(troi_dir)
            dest_fname_tpl = troi_dir + 'qxyz_regrid_{}_{}_{}.h5'.format(troiname,'{:06d}','{:06d}')

            for i in range(map_shape[0]):
                i_list = [i] * map_shape[1]
                j_list = range(map_shape[1]):
                total_datalength += map_shape[1]
                todo=[merged_fname,
                      raw_ds_path,
                      [dest_fname_tpl.format(i,j) for i,j in zip(i_list,j_list)],
                      troi,
                      troi_poni,
                      i_list,
                      j_list,
                      Q_dim,
                      troi_Theta_index_list,
                      troi_Theta_list,
                      fine_Theta_list,
                      [kappa, phi], 
                      verbose]
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
        pool.map_async(qrw.qxyz_regroup_employer,instruction_list)
        pool.close()
        pool.join()


    endreadtime = time.time()
    total_read_time = (endreadtime - starttime)
    print('='*25)
    print('\ntime taken for regrouping of {} datasets = {}'.format(total_datalength, total_read_time))
    print(' = {} Hz\n'.format(total_datalength/total_read_time))
    print('='*25) 
             
    result_fname_list = glob.glob(dest_dir+os.path.sep+'*.h5')

    if os.path.exists(Qmerged_fname):
        os.remove(Qmerged_fname)
    with h5py.File(Qmerged_fname) as dest_h5:
        q_group = dest_h5.create_group('entry/merged_data/Qxyz/')
        s_group = dest_h5.create_group('entry/merged_data/Sxys/')
        sum_group = dest_h5.create_group('entry/merged_data/sum')
        max_group = dest_h5.create_group('entry/merged_data/max')
        axes_group = dest_h5.create_group('entry/merged_data/axes')
        
        # get dtype, axis        
        with h5py.File(result_fname_list[0],'r') as source_h5:
            raw_ds_dtype = np.float64
            axes_dict = {'qx':'',
                         'qy':'',
                         'qz':''}
            
            for axis, ax_data in source_h5['entry/axes'].items():
                axes_group.create_dataset(name=axis,data=ax_data)
                if axis in axes_dict.keys():
                    axes_dict[axis]=np.asarray(ax_data)
            q_axes = [axes_dict['qx'], axes_dict['qy'], axes_dict['qz']]

                
        pointwise_group = q_group.create_group('single_files')
        print(Qdata_shape,raw_ds_dtype)
        q_all_ds = q_group.create_dataset(name='data_all', shape=Qdata_shape, dtype=raw_ds_dtype, compression='lzf')
        
        qx_ar = np.zeros(shape=map_shape, dtype = np.float64)
        qy_ar = np.zeros(shape=map_shape, dtype = np.float64)
        qz_ar = np.zeros(shape=map_shape, dtype = np.float64)
        q_ar = np.zeros(shape=map_shape, dtype = np.float64)
        
        sx_ar = np.zeros(shape=map_shape, dtype = np.float64)
        sy_ar = np.zeros(shape=map_shape, dtype = np.float64)
        sz_ar = np.zeros(shape=map_shape, dtype = np.float64)
        s_ar = np.zeros(shape=map_shape, dtype = np.float64)
        
        theta_ar = np.zeros(shape=map_shape, dtype = np.float64)
        phi_ar = np.zeros(shape=map_shape, dtype = np.float64)
        roll_ar = np.zeros(shape=map_shape, dtype = np.float64)
        pitch_ar = np.zeros(shape=map_shape, dtype = np.float64)
                        

        curr_qmax = np.zeros(shape=Qdata_shape[2:], dtype=raw_ds_dtype)
        curr_qsum = np.zeros(shape=Qdata_shape[2:], dtype=np.float64)
        
        curr_rsum = np.zeros(shape=map_shape, dtype=np.float64)
        curr_rmax = np.zeros(shape=map_shape, dtype=raw_ds_dtype)
        
        for fname in result_fname_list:
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
                    data_ij = np.asarray(dg['data'])
                    q_all_ds[r_i,r_j] = data_ij

                    curr_qmax = np.where(data_ij>curr_qmax,data_ij,curr_qmax)
                    curr_qsum += data_ij
                    curr_rmax[r_i,r_j] = np.float(dg['max'].value)
                    data_ij_sum = np.float(dg['sum'].value)
                    curr_rsum[r_i,r_j] = data_ij_sum

                    qx_ar[r_i,r_j] = np.float(dg['qx'].value)
                    qy_ar[r_i,r_j] = np.float(dg['qy'].value)
                    qz_ar[r_i,r_j] = np.float(dg['qz'].value)
                    q_ar[r_i,r_j] = np.float(dg['q'].value)

                    sx_ar[r_i,r_j] = np.float(dg['sx'].value)
                    sy_ar[r_i,r_j] = np.float(dg['sy'].value)
                    sz_ar[r_i,r_j] = np.float(dg['sz'].value)
                    s_ar[r_i,r_j] = np.float(dg['s'].value)

                    theta_ar[r_i,r_j] = np.float(dg['theta'].value)
                    phi_ar[r_i,r_j] = np.float(dg['phi'].value)
                    roll_ar[r_i,r_j] = np.float(dg['pitch'].value)
                    pitch_ar[r_i,r_j] = np.float(dg['roll'].value)


        qx_ds = q_group.create_dataset(name='qx', data = qx_ar)
        qy_ds = q_group.create_dataset(name='qy', data = qy_ar)
        qz_ds = q_group.create_dataset(name='qz', data = qz_ar)
        q_ds = q_group.create_dataset(name='q', data = q_ar)

        sx_ds = s_group.create_dataset(name='sx', data = sx_ar)
        sy_ds = s_group.create_dataset(name='sy', data = sy_ar)
        sz_ds = s_group.create_dataset(name='sz', data = sz_ar)
        s_ds = s_group.create_dataset(name='s', data = s_ar)

        theta_ds = q_group.create_dataset(name='theta', data = theta_ar * 180./np.pi)
        phi_ds = q_group.create_dataset(name='phi', data = phi_ar * 180./np.pi)
        roll_ds = q_group.create_dataset(name='roll', data = roll_ar * 180./np.pi)
        pitch_ds = q_group.create_dataset(name='pitch', data = pitch_ar * 180./np.pi)
        
        max_ds = max_group.create_dataset(name='q_space', data=curr_qmax)
        sum_ds = sum_group.create_dataset(name='q_space', data=curr_qsum)
        max_ds = max_group.create_dataset(name='r_space', data=curr_rmax)
        sum_ds = sum_group.create_dataset(name='r_space', data=curr_rsum)



    end_merge_time = time.time()
    total_merge_time = (end_merge_time - endreadtime)
    print('='*25)
    print('\ntime taken for merging of {} datasets = {}'.format(total_datalength, total_merge_time))
    print(' = {} Hz\n'.format(total_datalength/total_merge_time))
    print('='*25) 

    total_time = (end_merge_time - starttime)
    print('='*25)
    print('\ntotal time taken for {} datasets = {}'.format(total_datalength, total_time))
    print(' = {} Hz\n'.format(total_datalength/total_time))
    print('='*25) 
    

def main():

    merged_fname = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_rebin/merged.h5'

    Q_dim = [nQx, nQy, nQz] = [30,30,30]

    interp_factor = 5
    do_parallel_regrouping(merged_fname, Q_dim, interp_factor=interp_factor,limit_Thetas=True)

    
if __name__ == "__main__":
    main()

