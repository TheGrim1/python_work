from __future__ import print_function
import sys, os
import numpy as np

sys.path.append('/data/id13/inhouse2/AJ/skript') 

from fileIO.hdf5.frame_getter import master_getter
from fileIO.edf.save_edf import save_edf
from fileIO.hdf5.h5_tools import get_eigerrunno


def main(args):

    master_fname_list = [x for x in args if x.find('master.h5')>0]

    master_fname_list.sort()
    
    no_frames_per_output = 121   
    save_tpl = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-06-02_user_im22/PROCESS/aj_log/max_proj/theta_{:03d}_{:03d}.edf'
    

    for master_fname in master_fname_list:
        scan_no =  get_eigerrunno(master_fname)
        print(scan_no, master_fname)
        
        with master_getter(master_fname) as h5:
            if 'all_max' not in dir():
                all_max = np.zeros_like(h5[0])
            current = np.zeros_like(h5[0])
            out_index = 0
            for i,frame in enumerate(h5):
                current=np.where(frame>65000,current,np.where(frame>current,frame,current))
                print('on frame {} of {}'.format(i,os.path.basename(master_fname)))
                if type(no_frames_per_output)==type(None):
                    pass
                elif (i+1)%no_frames_per_output==0:
                    save_fname=save_tpl.format(scan_no,out_index)
                    out_index += 1
                    print('saving {}'.format(save_fname))
                    save_edf(current,save_fname)
                    all_max = np.where(current>all_max,current,all_max)
                    current = np.zeros_like(h5[0])
                
        save_edf(current,save_tpl.format(scan_no,out_index))
    save_edf(all_max,save_tpl.format(999,999))


            
    

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
    
    main(args)
