
import sys, os
import numpy as np
import fabio

sys.path.append('/data/id13/inhouse2/AJ/skript') 

import fileIO.hdf5.h5_tools as h5t


def _parse_motors_values_inner_outer(fname):
    basename = os.path.splitext(os.path.basename(fname))[0]
    bsplit = basename.split('_')
    prefix = '_'.join(bsplit[:-4])
    # return [prefix, bsplit[-4], float(bsplit[-3])/1000, bsplit[-2], float(bsplit[-1])/1000]
    return  [prefix, bsplit[-4], ((float(bsplit[-3])-180)%360.0)/1000, bsplit[-2], float(bsplit[-1])/1000]
    

def do_y_inner_outer_merge(args):

    todo_dict = {'sum':None,
                 'max':None,
                 'troi_ul':[[630,50],[100,100]],
                 'troi_ur':[[740,640],[100,100]],
                 'troi_lr':[[1520,950],[100,100]]
    }
    
    mapshape = (33,41)
    
    edf_fname_list = [on.path.realpath(x) for x in args if x.find('.edf')>0]

    parsed_list = [_parse_motors_values_inner_outer(x) for x in edf_fname_list])
    prefix = parsed_list[0][0]
    dest_fname = prefix + '_merged.h5'
    
    innermotor = parsed_list[0][3]
    outermotor = parsed_list[0][1]

    inner_pos_list = [x[4] for x in parsed_list]
    outer_pos_list = [x[2] for x in parsed_list]

    no_frames = len(inner_pos_list)
    if no_frames != mapshape[0]*mapshape[1]:
        print('invalid mapshape!')
        print(mapshape)
        print('frames found = {}'.format(len(no_frames)))

    # sort everything sensibly
        
    inner_pos_array = np.asaray(inner_pos_list).reshape(mapshape)
    outer_pos_array = np.asaray(outer_pos_list).reshape(mapshape)
    ind_i,ind_j = np.meshgrid(range(mapshape[1]),range(mapshape[0]))
    indexes = {}
    [indexes.update({x:[y[0],y[1]]}) for x,y in enumerate(zip(ind_i.flatten(),ind_j.flatten()))]
    
    inner_pos_1D = np.linspace(inner_pos_array.min(),inner_pos_array.max(),mapshape[0])
    outer_pos_1D = np.linspace(outer_pos_array.min(),outer_pos_array.max(),mapshape[1])
    
    with h5py.File(dest_fname,'w') as dest_h5:
        entry = dest_h5.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        axes_group = entry.create_group('axes')
        axes_group.attrs['NX_class'] = 'NXcollection'
        ax0 = axes_group.create_dataset(name=innermotor+'_1D', data=inner_pos_1D)
        ax1 = axes_group.create_dataset(name=outermotor+'_1D', data=outer_pos_1D)
        axes_group.create_dataset(name=innermotor, data=inner_pos_array)
        axes_group.create_dataset(name=outermotor, data=outer_pos_array)

        # setup groups and datasets:
        for job, info in todo_dict.items():
            group = entry.create_group(job)
            group.attrs('NX_class') = 'NXdata'
            group[innermotor]=ax0
            group[outermotor]=ax1
            if type(info) = type(None):
                group.create_dataset(name='data', shape=mapshape, dtype=int64)
            else:
                group.create_dataset(name='data', shape=list(mapshape)+info[1], dtype=int32)
                group.create_dataset(name='sum', shape=mapshape, dtype=int64)
                group.create_dataset(name='max', shape=mapshape, dtype=int64)
                
        for i,fname in enumerate(edf_fname_list):
            data = fabio.open(fname).data
            
                    
            
    

    

    
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

    # do_all_angles_split(args)
            
    do_y_inner_outer_merge(args)


