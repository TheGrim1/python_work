import sys, os
import h5py
import numpy as np
from nexusformat.nexus import *
import datetime
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import fileIO.hdf5.nexus_tools as nxt
import fileIO.hdf5.nexus_update_functions as nuf

class nx_id13:
    '''
    parent class to all id13 nexus dataformat saving implementations
    '''
    version = '0.0dev'

    '''to be extended to include the options:
    undefined, eh2_data_master, eh3_data_master, process_sas 
    as created by the corresponding child classes
    '''
    scantype = 'undefined'


    def __init__(self, fname ='/data/id13/inhouse6/nexustest_aj/new_master_design_class.h5'):
       
        ## each of these nx instances is fixed to one file
        self.fname   = fname

        
    def update_components(self, active_components = None):
        '''
        expects a dictionary with the info, eg:
        {'calibration':poni_fname,
         'Eiger4M'    :eiger_master_fname,
         'Vortex_1'   :xia_fname,
         'spec'       :spec_fname})
        depending on the further processing these keys may be used to create the corresponding links in the nexus file
        for the handling/ parsing see fileIO.hdf5.nexus_update_functions.update_group_from_file
        '''
        
        poni_fname         = None
        doolog_fname       = None
        eiger_master_fname = None
        spec_fname         = None
        samplename         = 'TODO, get this from eiger_master_fname'
        xia_fname          = None
        

        if active_components == None:
            self.active_components = {}
            self.active_components.update({'calibration'         :poni_fname,
                                           'Eiger4M'             :eiger_master_fname,
                                           'Vortex_1'            :xia_fname,
                                           'spec'                :spec_fname,
                                           'beamline_positioners':doolog_fname})

        else:
            pass
        
    def create_file(self):
        '''
        WARNING this overwrites self.fname
        '''
        fname = self.fname        
        self.nx_f = nx_f = nxload(fname, 'w')
        nx_f.attrs['file_name']        = fname
        nx_f.attrs['creator']          = 'nexus_class.py'
        nx_f.attrs['HDF5_Version']     = h5py.version.hdf5_version
        nx_f.attrs['NX_class']         = 'NXroot'
        nxt.timestamp(nx_f)
        
    def close_file(self):
        '''
        self.nx_f.close()
        '''
        ## Does not seem to impact the access to the file in self.nx_f = ?
        self.nx_f.close()

    def read_file(self):
        ''' 
        returns a read only version of self.nx_f
        '''
        self.close_file()
        return nxload(self.fname, 'r')

    
class nx_id13_eh2_data_master(nx_id13):
    '''
    parent: nx_id13
    This class handles the writing of the eh2 data master file
    '''
    scantype = 'eh2_data_master'

    def __init__(self,
                 fname ='/data/id13/inhouse6/nexustest_aj/new_master_design_class.h5',
                 active_components=None):
        
        ## each of these nx instances is fixed to one file
        self.fname      = fname
        self.prefix = 'TODO parse fname'
        self.update_components(active_components)
        
    def save_initial(self):
        '''
        saves the initial status of the beamline at the start of a scan
        for the handling/ parsing see fileIO.hdf5.nexus_update_functions.update_group_from_file
        '''
        
        nxentry = self.nx_f['entry'] = NXentry()

        ## setup instrument
    
        nxinstrument = nxentry['instrument'] = NXinstrument()
        nxinstrument.attrs['name'] = 'ID13_eh2'
    
        ## setup sample
        
        nxsample = nxentry['sample'] = NXsample()
        nxsample.attrs['prefix']       = self.prefix

        for (group, properties) in self.active_components.items():
            nx_instrument  = nuf.update_group_from_file(nxinstrument, group, properties)


        nxt.timestamp(self.nx_f)

        
    def do_test(self):
        self.create_file()
        self.save_initial()
