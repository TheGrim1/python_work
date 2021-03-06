import sys,os
import numpy as np
import h5py
from multiprocessing import Pool

sys.path.append('/data/id13/inhouse2/AJ/skript')
from simplecalc.colors import color_pixelarray

from fileIO.hdf5.h5_tools import get_eigerrunno, parse_master_fname
from simplecalc.slicing import troi_to_slice
        
def get_color_array_for_troi(troi):
    return color_pixelarray(troi[1])

def get_color_for_point(frame, color_array):
    if frame.sum()==0:
        return (0,0,0)
    else:
        return np.asarray(np.average(color_array.reshape((frame.shape[0]*frame.shape[1],3)),axis=0, weights=frame.flatten()),dtype=np.uint8)

def get_parasitic_color_for_point(frame,color_array):
    # set max picel 0
    np.ravel(frame)[np.argmax(frame)]*=0
    if frame.sum()==0:
        return (0,0,0)
    else:        
        return np.asarray(np.average(color_array.reshape((frame.shape[0]*frame.shape[1],3)),axis=0, weights=frame.flatten()),dtype=np.uint8)

    
    
def main(args):

    all_datafiles = [os.path.realpath(x) for x in args if x.find('.h5')>0]
    all_datafiles.sort()
    save_dir = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-29_inh_ihmi1397_aj/PROCESS/aj_log/analysis/merged'
    verbose = True
    bigmeshshape = (90*2+101, 500*7+561)
    y = np.arange(0.0,4.06,0.001)
    z = np.arange(0.0,1.124,0.004)
    troi = ((1335, 480), (40, 90))
    color_array=get_color_array_for_troi(troi)
    
    dest_fname = save_dir + os.path.sep + 'merged_exposed.h5'
    if os.path.exists(dest_fname):
        if verbose:
            print('removing {}'.format(dest_fname))

    print(all_datafiles)
    with h5py.File(dest_fname,'w') as dest_h5:
        entry = dest_h5.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        axes = entry.create_group('axes')
        axes.attrs['NX_class'] = 'NXcollection'
        axes.create_dataset('y',data=y)
        axes.create_dataset('z',data=z)
        axes.create_dataset('pxl_z',data=range(troi[0][0],troi[0][0]+troi[1][0]))
        axes.create_dataset('pxl_y',data=range(troi[0][1],troi[0][1]+troi[1][1]))
        axes.create_dataset('color_array',data=color_array)

        sum_group = entry.create_group('sum')
        sum_group['NX_class'] = 'NXdata'
        sum_ds = sum_group.create_dataset('data', shape=(list(bigmeshshape)), dtype=np.uint32)
        sum_ds.attrs['interpretation'] = u'image'
        
        data_group = entry.create_group('data')
        data_group.attrs['NX_class'] = 'NXdata'
        data = data_group.create_dataset('data', shape=(list(bigmeshshape)+list(troi[1])), dtype=np.uint32)
        data.attrs['interpretation'] = u'image'
        
        image_group = entry.create_group('image')
        image_group.attrs['NX_class'] = 'NXdata'
        image = image_group.create_dataset('data', shape=(list(bigmeshshape)+[3]), dtype=np.uint8)
        image.attrs['interpretation'] = u'image'
        
        parasitic_image_group = entry.create_group('parasitic_image')
        parasitic_image_group.attrs['NX_class'] = 'NXdata'
        parasitic_image = parasitic_image_group.create_dataset('data', shape=(list(bigmeshshape)+[3]), dtype=np.uint8)
        parasitic_image.attrs['interpretation'] = u'image'
        
        for source_fname in all_datafiles:
            print('working on {}'.format(source_fname))
            
            with h5py.File(source_fname,'r') as source_h5:
                source_data = source_h5['entry/data/data']
                source_sum = source_h5['entry/sum/data']

                indexes = np.where(np.asarray(source_sum)>1)
                for i,j in zip(indexes[0],indexes[1]):
                    data[i,j] = source_data[i,j]
                    sum_ds[i,j] = source_sum[i,j]
                    image[i,j] = get_color_for_point(source_data[i,j], color_array)
                    parasitic_image[i,j] = get_parasitic_color_for_point(source_data[i,j], color_array)
                    
        data_group['z'] = axes['z']
        data_group['y'] = axes['y']
        data_group['pxl_y'] = axes['pxl_y']
        data_group['pxl_z'] = axes['pxl_z']
        data_group.attrs['signal'] = 'data'
        data_group.attrs['axes'] = ['z','y','pxl_z','pxl_y']

        sum_group['z'] = axes['z']
        sum_group['y'] = axes['y']
        sum_group.attrs['signal'] = 'data'
        sum_group.attrs['axes'] = ['z','y']
        
        image_group['z'] = axes['z']
        image_group['y'] = axes['y']
        image_group.attrs['signal'] = 'data'
        image_group.attrs['axes'] = ['z','y']
        
        parasitic_image_group['z'] = axes['z']
        parasitic_image_group['y'] = axes['y']
        parasitic_image_group.attrs['signal'] = 'data'
        parasitic_image_group.attrs['axes'] = ['z','y']
        

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
