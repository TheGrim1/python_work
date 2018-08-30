import h5py
import numpy as np
import sys, os
import datetime
import time
from multiprocessing import Pool


from shutil import rmtree

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from pythonmisc.worker_suicide import worker_init
from fileIO.hdf5.workes import qxzy_regroup_worker as qrw
import pythonmisc.pickle_utils as pu
import pythonmisc.my_xrayutilities as my_xu

def parse_ij(fname):
    # tpl = Qxyz_redrid_{:06d}_{:06d}.h5
    i,j=[int(x) for x in os.path.splitext(fname)[0].split('_')[-2::]]
    return i,j

def do_parallel_regrouping(merged_fname, Q_dim, verbose = True):

    starttime = time.time()
    total_datalength = 0
    ### get the need info:
    no_processes=1
    with h5py.File(merged_fname,'r') as source_h5:
        troi_list = source_h5['entry/integrated/'].keys()
        Theta_list = asdf
        kappa = 0
        phi   = 0
        dest_dir = merged_fname+os.path.sep+'qxyz_regrouped/'
        if os.path.exists(dest_dir):
            print('deleting old data in {}'.format(dest_dir))
            rmtree(dest_dir)
        os.mkdir(dest_dir)

        for troiname in troi_list:
            troi_path = 'asdf/{}'.format(troiname)
            raw_ds_path = troi_path+'/asdf'
            raw_ds = source_h5[raw_ds_path]
            map_shape = list(raw_ds.shape[0:2])
            troi_poni = dict([[x,np.asarray(y)] for x,y in source_h5['entry/integrated/{}/calibration/{}/'.format(troiname)].items()])
            Qdata_shape = map_shape + Q_dim

            # find which Thetas are good
            if limit_Thetas:
                good_indexes = []
                sum_ds = source_h5[troi_path+'asdf']
                for i in range(sum_ds.shape[2]):
                    if sum_ds.sum(axis=0).sum(axis=0) > 100 *map_shape[0]*map_shape[1]: # counts are *1000
                        good_indexes.append(i)
                Theta_first = max(0,min(good_indexes)-1)
                Theta_last = min(len(Theta_list), max(good_indexes)+2)
                

            else:
                Theta_first = 0
                Theta_last = len(Theta_list)

            troi_Theta_list = Theta_list[Theta_first:Theta_last]
            print('found data between Theta_indexes {} and {}'.format(Theta_first, Theta_last))
            print('found data in Theta ',troi_Theta_list)
                
            # find Q regrouping, memory heavy!
            xu_exp = my_xu.get_id13_experiment(troi, troi_poni)
            # refine (interpolate) the  Theta list:
            fine_Theta_list = list(np.linspace(troi_Theta_list[0],troi_Theta_list[-1],(len(troi_Theta_list)-1)*interp_factor +1))
            qx, qy, qz = xu_exp.Ang2Q.area(fine_Theta_list,kappa,phi)


            # setup todolist for parrallel processes per map point
            troi_dir = (dest_dir+os.path.sep+troiname+os.path.sep)
            os.mkdir(troi_dir)
            dest_fname_tpl = troi_dir + 'Qxyz_regrid_{:06d}_{:06d}.h5'
            todo_list=[]
            for i in map_shape[0]:
                for j in map_shape[1]:
                    total_datalength += 1
                    todo=[merged_fname,
                          raw_ds_path,
                          dest_fname_tpl.format(i,j),
                          [i,j],
                          Q_dim,
                          troi_Theta_list,
                          fine_Theta_list,
                          [qx, qy, qz],
                          verbose]
                    
    print('setup parallel proccesses to write to {}'.format(dest_dir))
    instruction_list = []
    for i,todo in enumerate(todo_list):
        #DEBUG:
        instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
        instruction_list.append(instruction_fname)
    if no_processes==1:
        for instruction in instruction_list:
            qrw.qxyz_regroup_worker(instruction)
        ## non parrallel version for one dataset and timing:
        #fdw.fit_data_worker(instruction_list[0])
    else:
        pool = Pool(no_processes, worker_init(os.getpid()))
        pool.map_async(qrw.qxyz_regroup_employer,instruction_list)
        pool.close()
        pool.join()


    endtime = time.time()
    total_time = (endtime - starttime)
    print('='*25)
    print('\ntime taken for regrouping of {} datasets = {}'.format(total_datalength, total_time))
    print(' = {} Hz\n'.format(total_datalength/total_time))
    print('='*25) 
             
    result_fname_list = glob.glob(dest_dir+os.path.sep+'*.h5')

    if False:
        with h5py.File(merged_fname) as dest_h5:
            Q_group = dest_h5y.create_group('asdf')
            pointwise_group = q_group.create_group('single_files')
            Q_ds = Q_group.create_dataset(name='data',shape=Qdata_shape,dtype=raw_ds.dtype,compression='lzf')
            for fname in result_fname_list:
                print('collecting {}'.format(fname))
                i,j = parse_ij(fname)
                pointwise_group.create_dataset('point_{:06d}_{:06d}',data=fname)
                with h5py.File(fname,'r') as source_h5:
                    Q_ds[i,j] = np.asarray(source_h5['entry/data/data'])

    

def main():

    merged_fname = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_gpu2/merged_copy.h5'

    Q_dim = [nQx, nQy, nQz] = [10,10,10]
    
    do_parallel_regrouping(merged_fname, Q_dim)

    
if __name__ == "__main__":
    main()

