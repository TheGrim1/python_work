import h5py
import numpy as np
import click_tracker as ct
import matplotlib.pyplot as plt
from scipy.ndimage import shift as nd_shift
import data_io as data_io
import numpy as np
import datetime

def get_shift_from_coordfile(fname):

    coords, header = data_io.open_data(fname)
    
    shift = np.asarray([[[coords[0,0] - x[0],coords[0,1] - x[1]] for x in coords]][0])
    
    return shift



#    nd_shift(frame,shift, output = realigned_frame)
    
def link_axes(axes_group, link_group):

    for axis in axes_group.keys():
        link_group[axis]=axes_group[axis]
        

def shift_data(source_data, shift):

    data = np.rollaxis(source_data,-1)
    # print('shift.shape')
    # print(shift.shape)

    for i in range(shift.shape[0]):
        frame = np.zeros(shape = data[i].shape)
        nd_shift(data[i],shift[i],output=frame)
        data[i]=frame
    
    
    data = np.rollaxis(data,0,3)

    return data


class dest_h5group_writer(object):
    def __init__(self, dest_group):
        self.dest_group = dest_group
        
    def write(self, name, item):
        dest_group = self.dest_group
        if type(item) == h5py._hl.group.Group:
            # print('DEBUG, thinks its a group')
            # print('name:')
            # print(name)
            # print('item:')
            # print(item)
            # print('attrs:')
            
            new_group = dest_group.create_group(name)
            for attr_name, attr in item.attrs.items():
                new_group.attrs[attr_name] = attr

                # print(new_group.attrs[attr])
                
        elif type(item) == h5py._hl.dataset.Dataset:
            # print('DEBUG, thinks its a dataset')
            # print('name:')
            # print(name)
            # print('item:')
            # print(item)
            
            new_group = dest_group.create_dataset(name=name, data=np.asarray(item))

            for attr_name, attr in item.attrs.items():
                new_group.attrs[attr_name] = attr

def make_basefile(dest_h5, source_h5, groups_to_copy=['axes','coordinates','processing']):

    for group in groups_to_copy:
        dest_group = dest_h5.create_group(group)
        dest_group.attrs['NX_class'] = 'NXentry'
        dest_group_writer = dest_h5group_writer(dest_group)
        source_h5[group].visititems(dest_group_writer.write)


def append_manual_shift_to_processing(dest_h5, shift):
    if 'processing' in dest_h5.keys():
        processing_group = dest_h5['processing']
    else:
        processing_group = dest_h5.create_group('processing')
        processing_group.attrs['NX_class'] = 'NXentry'

    process_no = len(processing_group.keys())+1
    process_group = processing_group.create_group(name ='{}.manual_shift'.format(process_no))
    process_group.attrs['NXclass'] = 'NXprocess'
    process_group.attrs['date'] = str(datetime.datetime.now())
    process_group.attrs['program'] = str(__file__)
    process_group.attrs['sequence_index']=process_no
    process_group.attrs['version'] = '0.0.0'
    shift_ds = process_group.create_dataset(name='shift',data=shift)
    shift_ds.attrs['units']='pxl'
    
def make_realigned_spectrocrunch(dest_fname, source_fname, shift_datafname):

    shift = get_shift_from_coordfile(shift_datafname)
    
    with h5py.File(dest_fname,'w') as dest_h5:
        with h5py.File(source_fname,'r') as source_h5:
            make_basefile(dest_h5, source_h5)

            append_manual_shift_to_processing(dest_h5, shift)

            
            for entry in ['counters','detectorsum']:
                dest_entry_group = dest_h5.create_group(entry)
                dest_entry_group.attrs['NX_class']='NXdata'
                source_entry_group = source_h5[entry]
                
                for source_data_name, source_data_group in source_entry_group.items():
                    dest_data_group = dest_entry_group.create_group(source_data_name)
                    for attr_name, attr in source_data_group.attrs.items():
                        dest_data_group.attrs[attr_name] = attr
                    print('realigning {}'.format(source_data_name))
                    data = np.asarray(source_data_group['data'])

                    data = shift_data(data,shift)
                    new_ds = dest_data_group.create_dataset(name='data',data=data)

                    link_axes(dest_h5['axes'],dest_data_group)
                
                
                
    
