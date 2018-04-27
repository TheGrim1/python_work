import h5py
import fabio
import numpy as np
import sys, os
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

from fileIO.edf.open_edf import read_multiframe_edf
from fileIO.hdf5.h5_tools import parse_data_fname_tpl

def edf_to_master(source_fname, dest_master_fname, frames_per_file=2000):
    '''
    for laziness I write all the .edf(.gz) files into a structure similar to the Eiger datasaving format.
    the master file contains some detector info and a hard link to the data in entry/data/data_000001
    the data file contains only data in entry/data/data
    '''

    source_f = fabio.open(source_fname)

    data_fname_tpl = parse_data_fname_tpl(dest_master_fname)

    no_frames = source_f.nframes
    if no_frames % frames_per_file ==0:
        # exactly fitting no frames and frames per file
        no_datafiles = int(no_frames / frames_per_file)
    else:
        no_datafiles = int(no_frames / frames_per_file) +1
        
    frame_index_list = [range(x*frames_per_file,(x+1)*frames_per_file) for x in range(no_datafiles)]
    frame_index_list[-1] = range((no_datafiles-1)*frames_per_file, no_frames)
    
    datafiles = []
    for i, frame_indexes in enumerate(frame_index_list):
        data_fname= data_fname_tpl.format(i)
        with h5py.File(data_fname,'w') as data_dest_h5:
            datafiles.append(data_fname)
            #opening edfs takes ages!
            #data = read_multiframe_edf(source_fname, frame_indexes)
            datashape = (len(frame_indexes),source_f.data.shape[0],source_f.data.shape[1])
            datatype = source_f.data.dtype 
            data = np.zeros(shape=(datashape), dtype=datatype)
            for i, source_i in enumerate(frame_indexes):
                print('reading edf frame {} into data index{}'.format(source_i, i))
                print('data.sum() = {}'.format(source_f.data.sum()))
                source_f.currentframe = source_i
                data[i] = source_f.data
            # end read_multipleframe_edf
            
            data_dest_h5.create_group('entry')
            data_dest_h5['entry'].attrs['NX_class']='NXentry'
            data_dest_h5['entry'].create_group('data')
            data_dest_h5['entry/data'].attrs['NX_class']='NXdata'
            data_dest_h5['entry/data'].create_dataset(name = 'data', data=data)
            data_dest_h5.flush()

    with h5py.File(dest_master_fname,'w') as master_h5:
        master_h5.create_group('entry')
        master_h5['entry'].attrs['NX_class']='NXentry'
        master_h5['entry'].create_group('data')
        master_h5['entry/data'].attrs['NX_class']='NXdata'

        master_h5['entry'].create_group('instrument')
        master_h5['entry/instrument'].attrs['NX_class']='NXinstrument'
        master_h5['entry/instrument'].create_group('detector')
        master_h5['entry/instrument/detector'].attrs['NX_class']='NXdetector'
        master_h5['entry/instrument/detector'].create_dataset(name = 'x_pixel_size', data = 55e-6)
        master_h5['entry/instrument/detector'].create_dataset(name = 'y_pixel_size', data = 55e-6)
        master_h5['entry/instrument/detector'].create_dataset(name = 'description', data = 'ID01 ESRF maxipix')
        dataset_tpl = 'entry/data/data_{:06}'

        for i, data_fname in enumerate(datafiles):
            master_h5[dataset_tpl.format(i)]= h5py.ExternalLink(data_fname,'entry/data/data')
        master_h5.flush()
        
