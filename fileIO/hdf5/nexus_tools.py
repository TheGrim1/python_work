'''
This is supposed to turn into a tool that takes xyth data and images the strain as function of xy. Improved ofer h5_scan because this uses nexusformat.
Step 1 : 
* user imput of experiment parameter (file)

Step 2 : 
* validation (finding all files)
* collecting in one meta NXfile

<> independent:

Step 3 : integration
* fills the meta under NXfile/processing and /measurement

<> independent:

Step 4 : strain?
'''

import sys, os
import h5py
import numpy as np
from nexusformat.nexus import *
import datetime

# local import for testing:
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.hdf5.open_h5 import open_h5
import fileIO.hdf5.nexus_update_functions as nuf
import time

def create_nx_scan(fname = '/data/id13/inhouse6/nexustest_aj/new_master_design.h5'):

    nx_f = nxload(fname, 'w')
    nx_f.attrs['file_name']        = fname
    nx_f.attrs['creator']          = 'nexus_tools.py'
    nx_f.attrs['HDF5_Version']     = h5py.version.hdf5_version
    nx_f.attrs['NX_class']         = 'NXroot'
    timestamp(nx_f)
    
    return nx_f

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

def initialize_scan(nx_f):
    '''
    setup the empty group that will hopwfully be filled in the strain imaging workflow
What happens if a file is initialized 2x?
    '''
    ## required:
    poni_fname         = None
    doolog_fname       = None
    eiger_master_fname = None
    spec_fname         = None
    samplename         = 'TODO, get this from eiger_master_fname'
    xia_fname          = None
    
    nxentry = nx_f['entry'] = NXentry()

    ## setup instrument
    
    nxinstrument = nxentry['instrument'] = NXinstrument()
    nxinstrument.attrs['name'] = 'ID13_nano'
    
    # read beamline motor positions
    nxinstrument = nuf.update_motors_from_doolog(nxinstrument, doolog_fname)    

    ## setup sample

    nxsample = nxentry['sample'] = NXsample()
    nxsample.attrs['name']       = samplename

    ### components in the beamline:
    
    active_components = {}
    active_components.update({'calibration':poni_fname,
                              'Eiger4M'    :eiger_master_fname,
                              'Vortex_1'   :xia_fname,
                              'spec'       :spec_fname})
    
    for (group, fname) in active_components.items():
        nx_g = nxinstrument[group] = NXcollection()        ## maybe tis needs to be changed to NXdetector
        nx_g = nuf.update_group_from_file(nx_g, fname)


    timestamp(nx_f)
    
    return nx_f
