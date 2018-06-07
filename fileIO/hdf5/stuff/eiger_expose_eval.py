import numpy as np
import sys, os

import h5py # for the .h5 files
import fabio # for .edf files
from multiprocessing import Pool
import numpy as np
import glob

sys.path.append('/data/id13/inhouse2/AJ/skript/')
from fileIO.edf.edfIO import save_edf, open_edf
from fileIO.hdf5.h5_tools import parse_master_fname
from fileIO.datafiles.save_data import save_data
from fileIO.datafiles.open_data import open_data
from simplecalc.slicing import troi_to_slice

def single_datafile(args):
    
    master_fname = args[0]
    save_name_prefix = args[1]
    mask_fname = args[2]
    threshold = args[3]
    hist_range = np.asarray(args[4],dtype=np.uint64)
    data_key = args[5]
    verbose = args[6]
    test = args[7]


    data_sum = np.zeros(shape = (2167, 2070), dtype=np.uint64)
    data_max = np.zeros(shape = (2167, 2070), dtype=np.uint64)
    data_hist = np.zeros(shape = (len(hist_range)-1), dtype=np.uint64)
    data_count = np.zeros(shape = (len(hist_range), data_max.shape[0], data_max.shape[1]), dtype=np.uint64)

    no_frames = 0
    no_rejectframes = 0
    mask = fabio.open(mask_fname).data
    
    with h5py.File(master_fname,'r') as master_h5:
        
        data_path = 'entry/data/' + data_key
        print('pid: {}\n in file {}\n{}'.format(os.getpid(),master_fname,data_path))
        if test:
            data_group = master_h5[data_path][:2]
        else:
            data_group = master_h5[data_path]
        print data_group.shape
        for i, frame in enumerate(data_group):
            frame = np.asarray(np.where(mask,0,frame),dtype=np.uint64)
            # print(frame.dtype)
            if frame.max()>threshold:
                print('rejected frame {}'.format(i))
                no_rejectframes += 1
            else:
                print('kept frame {}'.format(i))
                data_sum += frame
                data_max = np.max((data_max,frame),axis=0)
                # print data_hist.shape
                # print np.asarray(np.histogram(frame, hist_range)[0], dtype=np.uint64).shape
                data_hist += np.asarray(np.histogram(frame, hist_range)[0], dtype=np.uint64)
                for count in hist_range[:-1]:
                    data_count[count] += np.asarray(np.where(frame==count,1,0), np.uint64)
                data_count[hist_range[-1]] += np.asarray(np.where(frame>=hist_range[-1],1,0), np.uint64)
                no_frames += 1

        hist_fname = save_name_prefix+data_key + '_hist_{}_taken_{}_rejected.txt'.format(no_frames, no_rejectframes)
        hist_data = np.asarray(zip(range(len(data_hist)),np.asarray(data_hist, dtype=np.uint64)))
        save_data(hist_fname, hist_data, header=['count_value','total_prevelance'])

        sum_fname = save_name_prefix+data_key+'_sum.edf'
        save_edf(data_sum, sum_fname)

        max_fname = save_name_prefix+data_key+'_max.edf'
        save_edf(data_max, max_fname)

        data_count = np.asarray(data_count, dtype=np.uint64)
        count_fname = save_name_prefix+data_key + '_noofcounts.edf'
        save_edf(data_count, count_fname)

    
def single_exposure(args):

    master_fname = args[0]
    save_name_prefix =args[1]
    mask_fname = args[2]
    threshold = args[3]
    verbose = args[4]
    test = args[5]
    
    hist_range = np.asarray(np.arange(threshold),dtype=np.uint64)

    data_sum = np.zeros(shape = (2167, 2070), dtype=np.uint64)
    data_max = np.zeros(shape = (2167, 2070), dtype=np.uint64)
    data_hist = np.zeros(shape = (len(hist_range)-1), dtype=np.uint64)
    data_count = np.zeros(shape = (len(hist_range), data_max.shape[0], data_max.shape[1]), dtype=np.uint64)
    no_frames = 0
    no_rejectframes = 0
    mask = fabio.open(mask_fname).data
    no_pixels = np.where(mask,0,1).sum()
    

    sub_save_name_prefix = save_name_prefix + os.path.sep
    if not os.path.exists(sub_save_name_prefix):
        os.mkdir(sub_save_name_prefix)
    
    if not os.path.exists(master_fname):
        print('file not found '.format(master_fname))

    with h5py.File(master_fname,'r') as h5_f:
        todo_list = []
        data_group = h5_f['entry/data']
        for key in data_group.keys():
            todo=[]
            todo.append(master_fname)
            todo.append(sub_save_name_prefix)
            todo.append(mask_fname)
            todo.append(threshold)
            todo.append(hist_range)
            todo.append(key)
            todo.append(verbose)
            todo.append(test)
            todo_list.append(todo)
            

    if test:
        single_datafile(todo_list[0])
    else:
        pool = Pool(processes=min(len(todo_list),20))
        pool.map_async(single_datafile,todo_list)
        pool.close()
        pool.join()
                        
                        
    if verbose:
        print('summed data saved in {}'.format(save_name_prefix))

        
    temp_files = glob.glob(sub_save_name_prefix+'*')

    print('found temp files:')
    print(temp_files)
    
    for fname in temp_files:
        if fname.find('_hist_')>0:
            data_hist += np.asarray(open_data(fname)[0][:,1],np.uint64)
            no_frames += int(fname.split('_')[-4])
            no_rejectframes += int(fname.split('_')[-2])
        if fname.find('_sum.edf')>0:
            data_sum += open_edf(fname)
        if fname.find('_max.edf')>0:
            data_max += open_edf(fname)
        if fname.find('_noofcounts.edf')>0:
            data_count += open_edf(fname)

    # saving full images:
    
    hist_fname = save_name_prefix+'_hist_{}_ppt_rejected.txt'.format(int(1000*no_rejectframes/(no_rejectframes+no_frames)))
    hist_data = np.asarray(zip(range(len(data_hist)),np.asarray(data_hist, dtype=np.float64)/no_frames/no_pixels))
    save_data(hist_fname, hist_data, header=['count','prevelance_per_frame'])

    avg = np.asarray(data_sum, dtype=np.float64)/no_frames
    avg_fname = save_name_prefix+'_avg.edf'
    save_edf(avg, avg_fname)

    max_fname = save_name_prefix+'_max.edf'
    save_edf(data_max, max_fname)

    data_count = np.asarray(data_count, dtype=np.float64)/no_frames
    for i, frame in enumerate(data_count):
        count_fname = save_name_prefix+'_freqof_{:03d}.edf'.format(i)
        save_edf(frame, count_fname)


    # saving the troi scanned with focussed beam:
    troi = ((1335, 480), (40, 90))
    cut = troi_to_slice(troi)
    save_name_prefix_list = save_name_prefix.split(os.path.sep)
    troi_path = os.path.sep.join(save_name_prefix_list[:-1]) + os.path.sep + 'troi_result'+os.path.sep
    
    if not os.path.exists(troi_path):
        os.mkdir(troi_path)
        
    troi_save_name_prefix = troi_path + save_name_prefix_list[-1]

    avg = np.asarray(data_sum[cut], dtype=np.float64)/no_frames
    avg_fname = troi_save_name_prefix+'_avg.edf'
    save_edf(avg, avg_fname)

    max_fname = troi_save_name_prefix+'_max.edf'
    save_edf(data_max[cut], max_fname)

    data_count = np.asarray(data_count, dtype=np.float64)/no_frames
    for i, frame in enumerate(data_count):
        count_fname = troi_save_name_prefix+'_freqof_{:03d}.edf'.format(i)
        save_edf(frame[cut], count_fname)
 
def main():

    args = ['/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-06_inh_ihsc1547_mro/DATA/AUTO-TRANSFER/eiger1/expose_detx790_1s_346_data_000003.h5',
            '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-06_inh_ihsc1547_mro/DATA/AUTO-TRANSFER/eiger1/expose_detx790_100ms_343_data_000003.h5',
            '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-06_inh_ihsc1547_mro/DATA/AUTO-TRANSFER/eiger1/expose_detx790_10ms_347_data_000003.h5',
            '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-06_inh_ihsc1547_mro/DATA/AUTO-TRANSFER/eiger1/expose_detx790_10ms_348_data_000003.h5',
            '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-06_inh_ihsc1547_mro/DATA/AUTO-TRANSFER/eiger1/expose_detx790_10ms_349_data_000003.h5',
            '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-06_inh_ihsc1547_mro/DATA/AUTO-TRANSFER/eiger1/expose_detx790_1p5ms_350_data_000003.h5']
            
    
    master_fnames = [parse_master_fname(x) for x in args if x.find('.h5')]
    
    save_path = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-06_inh_ihsc1547_mro/PROCESS/aj_log/analyse/'
    mask_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-06_inh_ihsc1547_mro/PROCESS/aj_log/mask.edf'

    threshold = [110,14,16,7,7,7,4]

    hist_range = range(101)
    
    verbose = True
    test = False
    todo_list = []
    for i,master_fname in enumerate(master_fnames):

        save_fname_prefix = save_path + os.path.basename(master_fname)[0:os.path.basename(master_fname).find('master')-1]
        todo=[]
        todo.append(master_fname)
        todo.append(save_fname_prefix)
        todo.append(mask_fname)
        todo.append(threshold[i])
        todo.append(verbose)
        todo.append(test)
        todo_list.append(todo)

    if test:
        single_exposure(todo_list[0])
    else:
        for todo in todo_list:
            single_exposure(todo)
    # single_exposure(todo_list[0])
        
        


    
if __name__=='__main__':

    
    main()
