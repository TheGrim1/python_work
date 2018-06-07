from __future__ import print_function
import sys

from fileIO.hdf5.frame_getter import master_getter
from fileIO.edf.save_edf import save_edf

def main():

    master_fname = '/mntdirect/_data_id13_inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/DATA/AUTO-TRANSFER/eiger1/setupb_25_master.h5'
    no_frames_per_output = 41
    
    save_tpl = '/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/PROCESS/logs/theta_scan/theta_{03d}.edf'
    

    with master_getter(master_fname) as h5:
        currrent = np.zeros_like(h5[0])
        for i,frame in enumerate(h5):
            current+=frame
            if i!=1 and (i-1)%no_frames_per_count==0:
                save_edf(current,save_tpl.format(i))
                current = np.zeros_like(h5[0])


            
    
if __name__ == '__main__':
    main()
