from __future__ import print_function
import sys, os
import numpy as np
import fabio

sys.path.append('/data/id13/inhouse2/AJ/skript') 
from silx.io.spech5 import SpecH5 as spech5
from fileIO.hdf5.frame_getter import data_getter
from fileIO.edf.save_edf import save_edf

import fileIO.hdf5.h5_tools as h5t
import fileIO.spec.spec_tools as st
from multiprocessing import Pool



def do_all_angles_worker(args):
    '''
    worker for do_all_angles_split
    '''

    master_fname = args[0]
    Theta_list = args[1]
    phi = float(args[2])%360.0

    dest_path = args[3]
    mask_fname= args[4]
    mask = np.where(fabio.open(mask_fname).data,0,1)
    no_frames_per_output = 101
    
    save_tpl = dest_path+os.path.sep+'all_angles_phi_{:06d}_theta_{:06d}.edf'
    ospid = os.getpid()
    with data_getter(master_fname) as h5:
        all_max = np.zeros_like(h5[0])
        current = np.zeros(shape=[no_frames_per_output,h5[0].shape[0],h5[0].shape[1]])
        out_index=0
        i = 0
        for frame in h5:
            current[i%no_frames_per_output] = frame
            print('process {} is on frame {}'.format(ospid,i))
            if (i+1)%no_frames_per_output==0:
                save_fname=save_tpl.format(int(phi*1000),int(Theta_list[out_index]*1000))
                out_index += 1
                print('process {} saving {}'.format(ospid,save_fname))
                curr_max = np.max(current,axis=0)
                curr_max *= mask
                save_edf(curr_max, save_fname)
                all_max = np.max([curr_max,all_max],axis=0)
                current*=0
            i+=1

    save_fname=save_tpl.format(int(phi*1000),99999)
    save_edf(all_max, save_fname)
    
    
def do_all_angles_split(data_fname_list):
    master_fname_list = [h5t.parse_master_fname(x) for x in data_fname_list if x.find('_data_')>0]
    master_fname_list = [x for x in master_fname_list if x.find('al2o3')<0]
    eigerrunno_list = [h5t.get_eigerrunno(x) for x in master_fname_list]

    r3_i_list = h5t.get_r3_i_list()

    eig_to_spec = {}
    [eig_to_spec.update({r3_i[1]:r3_i[2]}) for r3_i in r3_i_list]
    
    spec_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-06-23_inh_ihhc3435_aj/DATA/phi_kappa/phi_kappa.dat'
    dest_path = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-06-23_inh_ihhc3435_aj/PROCESS/aj_log/split_cumu'
    mask_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-06-23_inh_ihhc3435_aj/PROCESS/aj_log/mask.edf'
    
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    
    cumu_pars = []
    
    with spech5(spec_fname) as spec_f:
        for master_fname in master_fname_list:
            eigerrunno = h5t.get_eigerrunno(master_fname)
            if eigerrunno < 210:
                print(master_fname)

                scanno = eig_to_spec[eigerrunno]
                Theta_list = st.get_scan_motorpos(spec_f, scanno, 'Theta')[::101]
                phi = st.get_scan_motorpos(spec_f, scanno, 'smphi')
                cumu_pars.append([master_fname, Theta_list, phi, dest_path, mask_fname])
            
    # do_all_angles_worker(cumu_pars[0])
    
    pool = Pool(12)
    pool.map(do_all_angles_worker,cumu_pars)
    pool.close()
    pool.join()


def _y_inner_outer_worker(args):
    '''
    worker for do_all_angles_split
    '''

    master_fname = args[0]
    inner_list = args[1]
    no_frames_per_output = args[2]
    outer = float(args[3])%360.0
    dest_path = args[4]
    save_tpl = args[5]
    mask_fname= args[6]

    mask = np.where(fabio.open(mask_fname).data,0,1)
    
    ospid = os.getpid()
    with data_getter(master_fname) as h5:
        all_max = np.zeros_like(h5[0])
        all_sum = np.zeros(shape=h5[0].shape,dtype=np.int64)
        current = np.zeros(shape=[no_frames_per_output,h5[0].shape[0],h5[0].shape[1]],dtype=np.int32)
        out_index=0
        i = 0
        for frame in h5:
            current[i%no_frames_per_output] = frame
            print('process {} is on frame {}'.format(ospid,i))
            if (i+1)%no_frames_per_output==0:
                save_fname=save_tpl.format('{}',int(outer*1000),int(inner_list[out_index]*1000))
                out_index += 1
                print('process {} saving {}'.format(ospid,save_fname))
                curr_max = np.max(current,axis=0)
                curr_max *= mask
                save_edf(curr_max, save_fname.format('max'))
                all_max = np.max([curr_max,all_max],axis=0)

                curr_sum = np.sum(current,axis=0)
                curr_sum *= mask
                save_edf(curr_sum, save_fname.format('sum'))
                all_sum += curr_sum
                current*=0
            i+=1

    save_fname=save_tpl.format('{}',int(outer*1000),99999)
    save_edf(all_max, save_fname.format('max'))
    save_edf(all_sum, save_fname.format('sum'))
    
    
def do_y_inner_outer_split(data_fname_list):
    master_fname_list = [h5t.parse_master_fname(x) for x in data_fname_list if x.find('_data_')>0]
    master_fname_list = [x for x in master_fname_list if x.find('al2o3')<0]
    eigerrunno_list = [h5t.get_eigerrunno(x) for x in master_fname_list]

    r3_i_list = h5t.get_r3_i_list()

    eig_to_spec = {}
    [eig_to_spec.update({r3_i[1]:r3_i[2]}) for r3_i in r3_i_list]

    no_ypoints_dict = {}
    [no_ypoints_dict.update({int(r3_i[1]):int(r3_i[-1][1])}) for r3_i in r3_i_list]

    inner_motorname = 'Theta'
    outer_motorname = 'smphi'
    
    spec_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-06-23_inh_ihhc3435_aj/DATA/phi_kappa2/phi_kappa2.dat'
    dest_path = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-06-23_inh_ihhc3435_aj/PROCESS/aj_log/split_cumu/all_angles'
    mask_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-06-23_inh_ihhc3435_aj/PROCESS/aj_log/mask.edf'  

    save_tpl = dest_path+os.path.sep+'split_{}_{}_{}_{}_{}.edf'.format('{}',outer_motorname,'{:06d}',inner_motorname,'{:06d}')
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    
    cumu_pars = []

    print('inner_motorname = ' , inner_motorname)
    print('outer_motorname = ' , outer_motorname)
    print('spec_fname = ', spec_fname)
    print('dest_path = ', dest_path)
    print('mask_fname = ', mask_fname)

    print('master_fname_list: ')
    [print(x) for x in master_fname_list]
    

    
    with spech5(spec_fname) as spec_f:
        for master_fname in master_fname_list:
            eigerrunno = h5t.get_eigerrunno(master_fname)
            if eigerrunno >210:
                print('spec info: ')
                print(master_fname)
                scanno = eig_to_spec[eigerrunno]
                no_ypoints = no_ypoints_dict[eigerrunno]
                print('scanno = ', scanno)
                print('no_ypoints = ', no_ypoints)
                inner_list = st.get_scan_motorpos(spec_f, scanno, inner_motorname)[::no_ypoints]
                outer = st.get_scan_motorpos(spec_f, scanno, outer_motorname)
                print('outer = ', outer)
                print('inner_list = ', inner_list)
                
                cumu_pars.append([master_fname, inner_list, no_ypoints, outer, dest_path, save_tpl, mask_fname])
            
    # _yphi_kappa_worker(cumu_pars[0])
    
    pool = Pool(12)
    pool.map(_y_inner_outer_worker,cumu_pars)
    pool.close()
    pool.join()

    
                             
def main_old():

    master_fname = '/mntdirect/_data_id13_inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/DATA/AUTO-TRANSFER/eiger1/setupb_25_master.h5'
    no_frames_per_output = 41
    
    save_tpl = '/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/PROCESS/logs/theta_scan/theta_{:03d}.edf'
    

    with master_getter(master_fname) as h5:
        all_max = np.zeros_like(h5[0])
        current = np.zeros_like(h5[0])
        out_index = 0
        for i,frame in enumerate(h5):
            current=np.where(frame>65000,current,np.where(frame>current,frame,current))
            print(i)
            if (i+1)%no_frames_per_output==0:
                save_fname=save_tpl.format(out_index)
                out_index += 1
                print('saving {}'.format(save_fname))
                save_edf(current,save_fname)
                all_max = np.where(current>all_max,current,all_max)
                current = np.zeros_like(h5[0])

    save_edf(current,save_tpl.format(999))


            
## single scan:    
# if __name__ == '__main__':
#     main_old()


if __name__ == '__main__':
    
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc.  \n2)
python <thisfile.py> -f <file containing args as lines> \n3) find
<*yoursearch* -> arg1 etc.> | python <thisfile.py> """

    args = []
    if len(sys.argv) > 1:
        if sys.argv[1].find("-f")!= -1:
            f = open(sys.argv[2])
            for line in f:
                args.append(line.rstrip())
        else:
            args=sys.argv[1:]
    else:
        f = sys.stdin
        for line in f:
            args.append(line.rstrip())
    
    do_y_inner_outer_split(args)


