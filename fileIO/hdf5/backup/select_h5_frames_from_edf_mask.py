import sys, os
import numpy as np
import gc

sys.path.append('/data/id13/inhouse2/AJ/skript')

import fileIO.edf.open_edf as open_edf
import fileIO.hdf5.save_h5 as save_h5
import fileIO.hdf5.open_h5 as open_h5



def write_selected_h5(mask_fname, data_fname_tlp, frames_per_file, save_fname):
    

    mask = open_edf.open_edf(mask_fname)
    print mask.shape

    frame_list = []
    for ind, val in enumerate(mask.flatten()):
        if val>100:
            file_ind = ind / frames_per_file +1
            frame_ind = ind % frames_per_file
            data_fname = data_fname_tpl % file_ind
            frame_list.append([data_fname,[frame_ind]])
            print 'listed ',[data_fname,[frame_ind]]

    nframes_no = len(frame_list)
    print 'found %s frames' % nframes_no
    test_frame = open_h5.open_h5(frame_list[0][0],frame_list[0][1])
    print 'found single frame shape ', test_frame.shape
    shape = ([nframes_no] + list(test_frame.shape[1:]))
    print 'data.shape = ', shape 
    
    if np.asarray(shape).prod() > 2e8:
        # aleviate memory bottlenecks
        temp_file_fname = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/SESSION26/temp.tmp'
        print('created temp file: ',temp_file_fname) 
        data = np.memmap(temp_file_fname, dtype=np.int16, mode='w+', shape=tuple(shape))
    else:
        data = np.zeros(shape = shape)

    for i, frame_info in enumerate(frame_list):
        print 'reading frame %s of %s' %(i, nframes_no)
        data[i] = open_h5.open_h5(frame_info[0],frame_info[1])
                
    print 'saving data'
    save = save_h5.save_h5(data,save_fname)

    print 'doing cleanup'
    # neccessary cleanup for memmap
    if type(data) == np.core.memmap:
        tmpfname = data.filename
        del data
        gc.collect()
        os.remove(tmpfname)
            
if __name__=='__main__':

    # mask_fname = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/SESSION26/OUT_L2/a74_332_COF505_dry_heflushb__hit1_n_data__0000.edf'
    # data_fname_tpl = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/DATA/AUTO-TRANSFER/eiger4/COF505_dry_heflushb_332_data_%06d.h5'
    # frames_per_file = 500
    # save_fname = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/COF505_dry_heflusheb_332_hitfoundframes100_000001.h5'

    
    mask_fname = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/SESSION26/OUT_L2/a74_270_COF505_dry_heflush__hit1_n_data__0000.edf'
    data_fname_tpl = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/DATA/AUTO-TRANSFER/eiger4/COF505_dry_heflush_270_data_%06d.h5'
    frames_per_file = 500
    save_fname = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-06_inh_sc1481/PROCESS/aj_log/COF505_dry_heflushe_270_hitfoundframes100_000001.h5'




    
    write_selected_h5(mask_fname,
                      data_fname_tpl,
                      frames_per_file,
                      save_fname)
