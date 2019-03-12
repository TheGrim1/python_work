import h5py
import sys,os
import time
import glob
from shutil import rmtree
import numpy as np
from multiprocessing import Pool
import pprint

sys.path.append('/data/id13/inhouse2/AJ/skript')

from pythonmisc.worker_suicide import worker_init
from fileIO.hdf5.workers import bliss_fit_worker as ftw
import pythonmisc.pickle_utils as pu

def parse_ij(fname):
    # tpl = qxyz_redrid_{:06d}_{:06d}.h5
    i,j=[int(x) for x in os.path.splitext(fname)[0].split('_')[-2::]]
    troiname = os.path.splitext(fname)[0].split('_')[-3]
    return i,j

def do_fit(qmerged_fname, binning, verbose=False):

    fit_fname = os.path.dirname(qmerged_fname) + os.path.sep + 'fit_bin{}_'.format(binning)+os.path.basename(qmerged_fname)
    dest_dir = os.path.dirname(qmerged_fname)+ '/' + os.path.splitext(os.path.basename(fit_fname))[0] + '/'
    if os.path.exists(fit_fname):
        os.remove(fit_fname)
    if os.path.exists(dest_dir):
        print('deleting old data in {}'.format(dest_dir))
        rmtree(dest_dir)
    os.mkdir(dest_dir)
    starttime = time.time()
    total_datalength = 0  
    no_processes = 30
    
    
    with h5py.File(qmerged_fname,'r') as source_h5:
                        
        mask = np.asarray(source_h5['fluorescence/fluo_aligned/mask'])
        map_shape = tuple(mask.shape)
        mask[:2]*=0
        mask[-2:]*=0
        # setup todolist for parrallel processes per map point
        diff_g = source_h5['diffraction']
        todo_list = []
        troiname_list = []
        for troiname in diff_g.keys():
            troiname_list.append(troiname)

            dest_fname_tpl = dest_dir + 'fit_{}_{}_{}.h5'.format(troiname,'{:06d}','{:06d}')
            
            for i in range(map_shape[0]):
                i_list = [i] * map_shape[1]
                j_list = range(map_shape[1])
                total_datalength += map_shape[1]
                todo=[qmerged_fname,
                      dest_fname_tpl.format(i,j_list[0]),
                      troiname,
                      i_list,
                      j_list,
                      mask,
                      binning,
                      4]
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
            ftw.fit_worker(instruction)

        ## non parrallel version for one dataset and timing:
        #fdw.fit_data_worker(instruction_list[0])
    else:
        pool = Pool(no_processes, worker_init(os.getpid()))
        pool.map_async(ftw.fit_employer,instruction_list)
        pool.close()
        pool.join()


    endreadtime = time.time()
    total_read_time = (endreadtime - starttime)
    print('='*25)
    print('\ntime taken for fitting of {} datasets = {}'.format(total_datalength, total_read_time))
    print(' = {} Hz\n'.format(total_datalength/total_read_time))
    print('='*25) 
             
    result_fname_list = glob.glob(dest_dir+os.path.sep+'*.h5')


    with h5py.File(fit_fname,'w') as dest_h5:
        with h5py.File(qmerged_fname,'r') as qmerged_h5:
            print('writing to fit results file {}'.format(fit_fname))
            

            for troiname in troiname_list:
                troi_fname_list = [x for x in result_fname_list if x.find('_'+troiname+'_')>0]
                troi_g = dest_h5.create_group('results/'+troiname)

                qmerged_h5.copy('diffraction/{}/Q_masked'.format(troiname), dest_h5, 'results/'+troiname+'/analytical')

                fit_g = troi_g.create_group('fit')

                # first file to setup a containter for all data "fit_data_dict"
                with h5py.File(troi_fname_list[0],'r') as source_h5:
                    s_data =  source_h5['entry/data']
                    fit_data_dict={}

                    for group, members in s_data.items()[0][1].items():
                        print(members)
                        for name, member in members.items():
                            print(name)
                            print(member)
                            if type(member) == h5py._hl.group.Group:
                                parameters = member.attrs['parameters']
                                data_shape = tuple(list(map_shape)+[len(parameters)])
                                residual_shape = tuple(list(map_shape)+ list(member['residual'].shape))
                                fitname = '/'.join(member.name.split('/')[-2::])
                                print('fitname ' + fitname) 
                                fit_data_dict.update({fitname:
                                                      {'parameters':
                                                       parameters,
                                                       'peak0':np.zeros(shape=data_shape),
                                                       'peak1':np.zeros(shape=data_shape),
                                                       'residual':np.zeros(shape=residual_shape)}})

                print('found fit data:')
                print(fit_data_dict.keys())
                # fill the container with the data
                for fname in troi_fname_list:
                    # print('collecting {}'.format(fname))
                    f_i,f_j = parse_ij(fname)

                    with h5py.File(fname,'r') as source_h5:
                        
                        for i_j, dg in source_h5['entry/data'].items():
                            #print('collecting group {}'.format(i_j))
                            r_i,r_j = parse_ij(i_j)
                            frame_no = r_i*map_shape[1]+r_j

                            for fitname, fit_data in fit_data_dict.items():
                                keys = dg[fitname].keys()
                                
                                residual = np.asarray(dg[fitname]['residual'])
                                data_key = [x for x in keys if x!= 'residual'][0]
                                data = np.asarray(dg[fitname][data_key])
                                fit_data['peak0'][r_i,r_j] = data[0]
                                fit_data['peak1'][r_i,r_j] = data[1]
                                fit_data['residual'][r_i,r_j] = residual

                # write the collected data into the dest file:
                for fitname, fit_data in fit_data_dict.items():
                    ds_g = fit_g.create_group(fitname)
                    parameters = fit_data['parameters']
                    print('parameters')
                    print(parameters)
                    peak1 = fit_data['peak0']
                    peak2 = fit_data['peak1']
                    residual = np.asarray(fit_data['residual'])
                    ds_g.create_dataset(name='residual', data=residual)
                    ndim = residual.ndim
                    residual_sum = residual
                    for i in range(2,ndim):
                        residual_sum = residual_sum.sum(axis=-1)
                    ds_g.create_dataset(name='residual_sum', data=residual_sum)
                    for i, parameter in enumerate(parameters):
                        ds_g.create_dataset(name='peak0_' + parameter, data=peak1[:,:,i])
                        ds_g.create_dataset(name='peak1_' + parameter, data=peak2[:,:,i])
                        
                        

    endtime = time.time()
    total_time = (endtime - starttime)
    print('='*25)
    print('\ntotal time taken for fitting of {} datasets = {}'.format(total_datalength, total_time))
    print(' = {} Hz\n'.format(total_datalength/total_time))
    print('wrote to {}'.format(fit_fname))
    print('='*25)

             
        
def main():

    # qmerged_fname = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/previews/day_two/q15_int1_qxyz_kmap_and_cen_3b_merged.h5'
    qmerged_fname = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/previews/alignment/q23_int1_qxyz_kmap_rocking_merged.h5'

    pre_fit_binning = 3
    
    do_fit(qmerged_fname, binning=pre_fit_binning, verbose=True)

    
if __name__ == "__main__":
    main()
