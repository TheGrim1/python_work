import sys, os
import h5py
import numpy as np
import nexusformat.nexus as nx
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
        currently this creates the active components dictionary, in future the information will probably be parsed or passed on by some higher sources
        depending on the further processing these keys may be used to create the corresponding links in the nexus file
        for the handling/ parsing see fileIO.hdf5.nexus_update_functions.update_group_from_file
        '''

        #### This part could read from an external ini file:
        situation_dict     = {'name':'postion1'}
        calib_dict         = {'fname':None,
                              'situation':situation_dict}
        beamline_dict      = {'fname':None,
                              'instrument_name':self.instrument_name}
        
        eiger_dict         = {'fname':None,
                              'positioners':['detx',
                                             'dety',
                                             'detz']}
        spec_dict          = {'fname':None,
                              spec_scan_no_next:None}
        fluob_dict         = {'fname':None,
                              'positioners':['fluobx',
                                             'fluoby']}
        command_dict       = {'type'     :'eiger',
                              'version'  :'Eiger_3_Apples.py',
                              'cmd'      :'dscan dummy 0 1 1 0.1',
                              'motors'   :{'dummy':[0.0,1.0,1]},
                              'exp'      :0.1}
        samplename         = 'TODO, get this from eiger_master_fname'

        
        #### up to here

        #### the active components dict can be modified next:
        
        if active_components == None:
            self.active_components = {}
            self.active_components.update({'calibration'         :calib_dict,
                                           'Eiger4M'             :eiger_dict,
                                           'fluob'               :fluob_dict,
                                           'spec'                :spec_dict,
                                           'beamline_positioners':beamline_dict,
                                           'inital_command'      :command_dict})

        else:
            print 'TODO get these components form some configuration file'
            
        
    def create_file(self):
        '''
        WARNING this overwrites self.fname
        '''
        fname = self.fname        
        self.nx_f = nx_f = nx.nxload(fname, 'w')
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

    def change_to_read_only(self):
        ''' 
        returns a read only version of self.nx_f
        '''
        self.close_file()
        return nx.nxload(self.fname, 'r')

    def save_initial(self, scan_command = 'dmesh '):
        '''
        saves the initial status of the beamline at the start of a scan
        for the handling/ parsing see fileIO.hdf5.nexus_update_functions.update_group_from_file
        '''        
        nxentry = self.nx_f['entry'] = nx.NXentry()

        ## setup instrument
    
        nxinstrument = nxentry['instrument'] = nx.NXinstrument()
        nxinstrument.attrs['name'] = self.instrument_name
    
        ## setup sample
        
        nxsample = nxentry['sample'] = nx.NXsample()
        nxsample.attrs['prefix']       = self.prefix

        for (group, properties) in self.active_components.items():
            nxinstrument  = nuf.update_group_from_file(nxinstrument, group, properties)

        print self.nx_f.tree

        self.close_file()
        print self.nx_f.tree
        self.set_default_links()
            
        nxt.timestamp(self.nx_f)

    def set_default_links(self, default_detector = 'Eiger4M'):
        '''
        links entry/data/data to the default detector
        '''
        data = self.get_detector_link(default_detector)

        axes = self.get_scanned_axes_names()

        for i, axisname in enumerate(axesnames):
            axis = nx.NXfield(self.get_motor_link(axisname), name = axisname)
            axes.append(axis)
        
        nxdata = self.nx_f['entry'] = nx.NXdata(nx.NXfield(data, name = 'data'), axes)
        
    def get_scanned_axes_names(self):
        '''
        returns list with the neames of the scanned axes
        '''
        axes = []
        for axis in self.nx_f['entry/instrument/beamline_positioners/positioners_initial/scan']:
            axis.append(axis.name)
        return axes
        
    def get_detector_link(self, detector_name):
        '''
        returns the data of <detector_name>
        '''
        detectorpath = 'entry/instrument/'+ detector_name + '/data'
        if type(self.nx_f[detectorpath]) == nx.NXlink:
            return detectorpath
        else:
            return nx.NXlink(self.nx_f['detectorpath'])

    def get_motor_link(self, motor_name):
        '''
        returns a link to the best guess of where the latest motor position may be
        '''
        blp = self.nx_f['entry/instrument/beamline_positioners']

        if 'positioners_final' in blp.keys():
            blp_final = blp['positioners_final']
            nx_link = nx.NXlink(blp_final[nxt.find_dataset_path(blp_final, motor_name)])
        else:
            nx_link = nx.NXlink(self.nx_f['entry/instrument/initial_commmand/' + motor_name])
                            
        return nx_link
        
    def do_test(self):
        self.create_file()
        self.save_initial()

    def insert_processed_data(self,
                              data  = np.random.randn(5,10,7),
                              process = 'add_random_data',
                              **kwargs):
        """
        arguments:
        name  = 'random_counts'
        axes  = [('x',{'units':'mm','values':np.arange(5)}),
        ('time',{'values': np.arange(10)}),
        ('thumbs',{'units':'mm','values':np.arange(7)})]
        # for now links only work if they exist :(
        """

        
        if 'entry' in self.nx_f.keys():
            nx_entry =  self.nx_f['entry']
        else:
            nx_entry =  self.nx_f['entry'] = nx.NXentry()
            
        i = 0
        process_tpl = process +'_%04d'
        while process_tpl % i in nx_entry.keys():
            i += 1
            if i>999:
                raise ValueError('max process number reached')
        if i :
            process = process_tpl % i
        
        nx_g = nx_entry[process] = nx.NXdata()
            
        nx_g = nuf.insert_dataset(nx_g,
                                  data = data,
                                  **kwargs)
        

class nx_id13_eh2_data_master(nx_id13):
    '''
    parent: nx_id13
    This class handles the writing of the eh2 data master file
    '''
    scantype = 'eh2_data_master'

    def __init__(self,
                 fname = "/data/id13/inhouse6/nexustest_aj/pre_master.h5",
                 active_components=None):

        ## each of these nx instances is fixed to one file
        self.fname      = fname
        self.prefix     = 'TODO parse from fname'
        self.instrument_name = 'id13_eh2'
        self.update_components(active_components)

            
class nx_id13_eh3_data_master(nx_id13):
    '''
    parent: nx_id13
    This class handles the writing of the eh2 data master file
    '''
    scantype = 'eh2_data_master'
    nx_data = nx.NXdata

    def __init__(self,
                 fname = "/data/id13/inhouse6/nexustest_aj/pre_master.h5",
                 active_components=None):

        ## each of these nx instances is fixed to one file
        self.fname      = fname
        self.prefix     = 'TODO parse from fname'
        self.instrument_name = 'id13_eh3'
        self.update_components(active_components)

        

