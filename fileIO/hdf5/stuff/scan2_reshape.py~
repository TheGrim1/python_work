import sys,os
import numpy as np
import h5py
from multiprocessing import Pool

sys.path.append('/data/id13/inhouse2/AJ/skript/')

from fileIO.hdf5.h5_tools import get_eigerrunno, parse_master_fname
from simplecalc.slicing import troi_to_slice
import fabio        

def reshape_worker(args):
    
    data_fname = args[0]
    save_dir = args[1]
    bkg_fname = args[2]
    verbose = args[3]

    y = np.arange(0.0,4.06,0.001)
    z = np.arange(0.0,1.124,0.004)
    troi = ((1335, 480), (40, 90))
    
    sourcemaster_fname = parse_master_fname(data_fname)
    eigerrunno = get_eigerrunno(sourcemaster_fname)
    scan_row = int((eigerrunno-25)/8)
    scan_col = (eigerrunno-25)%8
      
    y_offset = scan_col*0.5
    z_offset = -scan_row*0.36
    yi_offset = 560 + scan_col*500
    zi_offset = 180 - scan_row*90

    yi,zi = np.meshgrid(range(yi_offset,yi_offset-560,-1),range(zi_offset,zi_offset+101,1))
    li,ki = np.meshgrid(range(560),range(101))
    indexes = zip(ki.flatten(),li.flatten(),zi.flatten(),yi.flatten())

    bigmeshshape = (90*2+101, 500*7+561)
    dest_fname = save_dir +os.path.sep+ 'reshaped_'+os.path.basename(data_fname)

    normalization = np.asarray(fabio.open(bkg_fname).data)
    pid=os.getpid()
    if os.path.exists(dest_fname):
        if verbose:
            print('removing {}'.format(dest_fname))
        
        os.remove(dest_fname)

    if verbose:
        print('pid: {} reading {}'.format(pid, sourcemaster_fname))
              
          
    with h5py.File(dest_fname,'w') as dest_h5:
        entry = dest_h5.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        axes = entry.create_group('axes')
        axes.attrs['NX_class'] = 'NXcollection'
        axes.create_dataset('y',data=y)
        axes.create_dataset('z',data=z)
        axes.create_dataset('pxl_z',data=range(troi[0][0],troi[0][0]+troi[1][0]))
        axes.create_dataset('pxl_y',data=range(troi[0][1],troi[0][1]+troi[1][1]))

        sum_group = entry.create_group('sum')
        sum_group.attrs['NX_class'] = 'NXdata'
        sum_ds = sum_group.create_dataset('data', shape=(list(bigmeshshape)), dtype=np.uint64)
        sum_ds.attrs['interpretation'] = u'image'
        
        data_group = entry.create_group('data')
        data_group.attrs['NX_class'] = 'NXdata'
        data = data_group.create_dataset('data', shape=(list(bigmeshshape)+list(troi[1])), dtype=np.uint32)
        data.attrs['interpretation'] = u'image'

        frame_no = 0
        with h5py.File(sourcemaster_fname,'r') as source_h5:
            source_group = source_h5['entry/data']
            source_keys = source_group.keys()
            source_keys.sort()
            for key in source_keys:
                for frame in source_group[key]:
                    if frame_no%100==0:
                        print('pid: {} file: {} frame: {}'.format(pid, os.path.basename(sourcemaster_fname),frame_no))
                    k,l,i,j = indexes[frame_no]

                    if normalization[k,l] == 0:
                        frame_roi = np.zeros(shape=troi[1],dtype=np.uint64)
                    else:
                        frame_roi = np.asarray((frame[troi_to_slice(troi)]/normalization[k,l]),dtype=np.uint64)
                    data[i,j] = frame_roi
                    sum_ds[i,j] = frame_roi.sum()

                    frame_no+=1

        data_group['z'] = axes['z']
        data_group['y'] = axes['y']
        data_group['pxl_y'] = axes['pxl_y']
        data_group['pxl_z'] = axes['pxl_z']
        data_group.attrs['signal'] = ['data']
        data_group.attrs['axes'] = ['z','y','pxl_z','pxl_y']

        sum_group['z'] = axes['z']
        sum_group['y'] = axes['y']
        sum_group.attrs['signal'] = ['data']
        sum_group.attrs['axes'] = ['z','y']

      
def main(args):

    all_datafiles = [os.path.realpath(x) for x in args if x.find('.h5')>0][:-1]
    no_processes = len(all_datafiles)
    
    spec_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-29_inh_ihmi1397_aj/DATA/align/align.dat'
    save_dir = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-29_inh_ihmi1397_aj/PROCESS/aj_log/analysis/reorderd'
    bkg_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-29_inh_ihmi1397_aj/PROCESS/aj_log/hitmaps/norm_on_70perc.edf'
    verbose = True

    todo_list = []
    
    for i, data_fname in enumerate(all_datafiles):
        todo = []
        todo.append(data_fname)
        todo.append(save_dir)
        todo.append(bkg_fname)
        todo.append(verbose)
        todo_list.append(todo)

    print('todo_list')
    print(todo_list)

    # reshape_worker(todo_list[0])
    
    pool = Pool(processes=no_processes)
    pool.map_async(reshape_worker, todo_list)
    pool.close()
    pool.join()
     
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
