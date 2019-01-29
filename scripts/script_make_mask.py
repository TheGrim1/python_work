USAGE = ''' python script_make_mask.py <master_fname> <optional percentile threshold>
\n works best on flatfield type data\n can be used with calib scans if they are sufficiently sparse'''

import h5py
import sys,os
import numpy as np
sys.path.append('/data/id13/inhouse2/AJ/skript')
from fileIO.edf.save_edf import save_edf

def main(args):
    master_fname = os.path.realpath(args.pop(0))
    if len(args) > 0:
        threshold= args.pop(0)
    else:
        threshold = 100

    save_fname = os.getcwd() + os.path.sep  +os.path.basename(master_fname)[:os.path.basename(master_fname).find('master')] + 'mask.edf'
    saveneg_fname = os.getcwd() + os.path.sep  +os.path.basename(master_fname)[:os.path.basename(master_fname).find('master')] + 'mask_neg.edf'
        

    with h5py.File(master_fname,'r') as m5:
        datalength = m5['entry/data/data_000001'].shape[0]
        no_frames = np.min([datalength,1000])
        print('reading {} frames in:\n{}'.format(no_frames, master_fname))
        data = np.asarray(m5['entry/data/data_000001'][:no_frames])

    print('mask == 1 where 75 percetile above thereshold {}'.format(threshold))
    data_perc = np.percentile(data,75,axis=0)

    mask = np.asarray(np.where(data_perc>threshold,1,0),dtype=np.uint8)
    mask_neg = np.where(mask,0,1)

    print('saving mask as:\n'+save_fname)
    save_edf(mask,save_fname)
    save_edf(mask_neg,saveneg_fname)


if __name__=='__main__':
    args = sys.argv[1:]
    if len(args) not in [1,2]:
        print(USAGE)
    else:
        main(args)
