import sys, os
import h5py
import numpy as np
from nexusformat.nexus import *
import datetime

# local import for testing:
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.hdf5.open_h5 import open_h5
import time


def timestamp(nx_f = None):
    '''
    timestamps the passed nexus file, returns 1 if succesfull, -1 else
    '''
    if type(nx_f) == h5py._hl.files.File or type(nx_f) == NXroot:

        timestamp = "T".join(str(datetime.datetime.now()).split())
        if 'file_time' in nx_f.attrs.keys():
            nx_f.attrs['file_update_time'] = timestamp
        else:
            nx_f.attrs['file_time']        = timestamp
        test = 1
    else:
        test = -1
    return test


def find_dataset_path(nx_g, dataset_name):
    '''
    returns the path to dataset_name within the groups in nx_g.
    kind of like to find --maxdepth=1
    '''
    dataset_path = 'did not find a valid path'
    for key in nx_g.keys():
        for dataset in nx_g[key]:
            if dataset.name == dataset_name:
                data_set.path = key + '/' + dataset_name
    
    return dataset_path


def id13_default_units(name):
    angles = ['Theta',
              'Rot1',
              'Rot2',
              'Rot3']
    piezo  = ['nnp1',
              'nnp2',
              'nnp3']
    time   = ['time',
              'exp']
    meter  = ['PixelSize1',
              'PixelSize2',
              'Distance',
              'Poni1',
              'Poni2',
              'Wavelength']
    
    if name in angles:
        units = 'degrees'
    elif name in meter:
        units = 'm'
    elif name in piezo:
        units = 'um'
    elif name in time:
        units = 's'
        
    else:
        units = 'mm'
    return units
