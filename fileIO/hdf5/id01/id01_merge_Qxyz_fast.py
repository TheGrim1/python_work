import h5py
import sys,os
import time
import glob
from shutil import rmtree
import numpy as np

sys.path.append('/data/id13/inhouse2/AJ/skript')
import fileIO.spec.spec_tools as st
from pythonmisc.worker_suicide import worker_init
from fileIO.hdf5.workers import id01qxyz_regroup_worker_fast as qrw
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import rebin, troi_to_slice
from scipy.ndimage.measurements import center_of_mass as com   
from multiprocessing import Pool


def calc_sd(data, data_sum, COM, axes):
    '''
    shape data has to be 'shape' of the list axes
    data.ndim == len(COM) == len(axes)
    COM is the index of the com, data_sum its mass
    return sx, sy, sz, s
    '''
    weights = np.meshgrid(*np.stack([axes[i]-COM[i] for i in range(3)]))

    weighted_data = (data * weights) 
    
    sx, sy, sz = [((weighted_data[i]**2).sum()/data_sum)**0.5 for i in range(3)]
    
    return np.asarray([sx, sy, sz])

def parse_ij(fname):
    # tpl = qxyz_redrid_{:06d}_{:06d}.h5
    i,j=[int(x) for x in os.path.splitext(fname)[0].split('_')[-2::]]
    return i,j

def do_regrouping(merged_fname, Q_dim, map_shape, par_dict, interp_factor, prefix='', verbose = True):
    Qmerged_fname = os.path.dirname(merged_fname) + os.path.sep + prefix + 'qxyz_'+os.path.basename(merged_fname)
    starttime = time.time()
    total_datalength = 0    ### get the need info:
    no_processes = 10

    bin_size = par_dict['bin_size']
    
    with h5py.File(merged_fname,'r') as source_h5:
        
        dest_dir = os.path.dirname(merged_fname)+os.path.sep+prefix+'qxyz_regrouped/'
        if os.path.exists(dest_dir):
            print('deleting old data in {}'.format(dest_dir))
            rmtree(dest_dir)
        os.mkdir(dest_dir)

        ### change troi size to rebin, required for qxyz
        troi = par_dict['troi']
        troi = [troi[0],[int(troi[1][0]/bin_size),int(troi[1][1]/bin_size)]]
            
        Qdata_shape = tuple(list(map_shape) + list(Q_dim))
            
        # setup todolist for parrallel processes per map point

        dest_fname_tpl = dest_dir + 'qxyz_regrid_{}_{}.h5'.format('{:06d}','{:06d}')
        todo_list = []
        for i in range(map_shape[0]):
            i_list = [i] * map_shape[1]
            j_list = range(map_shape[1])
            total_datalength += map_shape[1]
            todo=[merged_fname,
                  [dest_fname_tpl.format(i,j) for i,j in zip(i_list,j_list)],
                  i_list,
                  j_list,
                  Q_dim,
                  map_shape,
                  par_dict,
                  interp_factor,
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
            qrw.id01qxyz_regroup_worker(instruction)

        ## non parrallel version for one dataset and timing:
        #fdw.fit_data_worker(instruction_list[0])
    else:
        pool = Pool(no_processes, worker_init(os.getpid()))
        pool.map_async(qrw.id01qxzy_regroup_employer,instruction_list)
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
            raw_ds_dtype = source_h5['entry/data/data'].dtype
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

        
        curr_qmax = np.zeros(shape=Qdata_shape[2:], dtype=raw_ds_dtype)
        curr_qsum = np.zeros(shape=Qdata_shape[2:], dtype=np.float64)
        
        curr_rsum = np.zeros(shape=map_shape, dtype=np.float64)
        curr_rmax = np.zeros(shape=map_shape, dtype=raw_ds_dtype)
        
        for fname in result_fname_list:
            print('collecting {}'.format(fname))
            i,j = parse_ij(fname)
            pointwise_group.create_dataset('point_{:06d}_{:06d}'.format(i,j),data=fname)
            with h5py.File(fname,'r') as source_h5:

                # raw data:
                data_ij = np.asarray(source_h5['entry/data/data'])
                q_all_ds[i,j] = np.asarray(source_h5['entry/data/data'])
                
                curr_qmax = np.where(data_ij>curr_qmax,data_ij,curr_qmax)
                curr_qsum += data_ij
                curr_rmax[i,j] = data_ij.max()
                data_ij_sum = data_ij.sum()
                curr_rsum[i,j] = data_ij_sum

                # center of mass, var
                i_COM = com(data_ij)
                qx, qy, qz = q_com = np.asarray([np.interp(x,range(len(q_axes[i])),q_axes[i]) for i,x in enumerate(i_COM)])
                q = (q_com**2).sum()**0.5
                sx, sy, sz = sigma = calc_sd(data_ij, data_ij_sum, q_com, q_axes)
                s = (sigma**2).sum()**0.5

                qx_ar[i,j] = qx
                qy_ar[i,j] = qy
                qz_ar[i,j] = qz
                q_ar[i,j] = q

                sx_ar[i,j] = sx
                sy_ar[i,j] = sy
                sz_ar[i,j] = sz
                s_ar[i,j] = s

                # angles
                theta_ar[i,j] = np.arccos(qz/q)
                phi_ar[i,j] = np.arctan(qx/qy)


        qx_ds = q_group.create_dataset(name='qx', data = qx_ar,compression='lzf')
        qy_ds = q_group.create_dataset(name='qy', data = qy_ar,compression='lzf')
        qz_ds = q_group.create_dataset(name='qz', data = qz_ar,compression='lzf')
        q_ds = q_group.create_dataset(name='q', data = q_ar,compression='lzf')

        sx_ds = s_group.create_dataset(name='sx', data = sx_ar,compression='lzf')
        sy_ds = s_group.create_dataset(name='sy', data = sy_ar,compression='lzf')
        sz_ds = s_group.create_dataset(name='sz', data = sz_ar,compression='lzf')
        s_ds = s_group.create_dataset(name='s', data = s_ar,compression='lzf')

        theta_ds = q_group.create_dataset(name='theta', data = theta_ar * 180./np.pi,compression='lzf')
        phi_ds = q_group.create_dataset(name='phi', data = phi_ar * 180./np.pi,compression='lzf')
        
                
        max_ds = max_group.create_dataset(name='q_space', data=curr_qmax,compression='lzf')
        sum_ds = sum_group.create_dataset(name='q_space', data=curr_qsum,compression='lzf')
        max_ds = max_group.create_dataset(name='r_space', data=curr_rmax,compression='lzf')
        sum_ds = sum_group.create_dataset(name='r_space', data=curr_rsum,compression='lzf')



    end_merge_time = time.time()
    total_merge_time = (end_merge_time - endreadtime)
    print('='*25)
    print('\ntime taken for merging of {} datasets = {}'.format(total_datalength, total_merge_time))
    print(' = {} Hz\n'.format(total_datalength/total_merge_time))
    print('='*25) 

    
    endtime = time.time()
    total_time = (end_merge_time - starttime)
    print('='*25)
    print('\ntotal time taken for {} datasets = {}'.format(total_datalength, total_time))
    print(' = {} Hz\n'.format(total_datalength/total_time))
    print('='*25) 
             

def main():

    merged_fname = '/data/id13/inhouse2/AJ/data/ma3576/id01/analysis/dose_mica/dose_mica.h5'
    

    map_shape = [100,100]
    # troi = [[261,106],[160,320]]
    troi = [[0,0],[512,512]]
    # troi = [[0,0],[10,10]]
    bin_size = 1
    interp_factor = 1
    Q_disc = 50
    Q_dim = [nQx, nQy, nQz] = [Q_disc]*3
    
    prefix = 'q{}_bin{}_int{}_'.format(Q_disc,bin_size,interp_factor)
    
    par_dict = {'cch1':350.0,
                'cch2':350.5,
                'distance':570.8,
                'pixel_width':0.055,
                'troi':troi,
                'bin_size':bin_size,
                'energy_keV':8.0}

    # check size of rebinned data:
    dummy=np.empty(shape=(4000,4000),dtype=np.uint8)
    Nch1, Nch2 = rebin(dummy[troi_to_slice(troi)],[bin_size]*2).shape
    par_dict['Nch1'] = Nch1
    par_dict['Nch2'] = Nch2
    
    

    do_regrouping(merged_fname, Q_dim, map_shape, par_dict, interp_factor=interp_factor, prefix=prefix, verbose=True)

    
if __name__ == "__main__":
    main()
