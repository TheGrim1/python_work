from __future__ import print_function
import sys
import numpy as np

sys.path.append('/data/id13/inhouse2/AJ/skript') 

from fileIO.hdf5.frame_getter import master_getter
from fileIO.edf.save_edf import save_edf

def main():

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


            
    
if __name__ == '__main__':
    main()
