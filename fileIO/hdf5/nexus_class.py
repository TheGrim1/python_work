
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
    undefined, eh2_premaster, eh3_premaster, process_sas, eh2_postmaster
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
        sample_dict        = {'auto_update':True,
                              'sample_name':'Teddy v5'}
        calib_dict         = {'fname':None,
                              'situation':'pos1'}
        beamline_dict      = {'fname':None,
                              'instrument_name':self.instrument_name}
        
        eiger_dict         = {'fname':None,
                              'positioners':['detx',
                                             'dety',
                                             'detz']}
        spec_dict          = {'fname':None,
                              'spec_scan_no_next':None}
        xia_dict           = {'fname':None,
                              'positioners':['fluobx',
                                             'fluoby']}
        command_dict       = {'type'     :'eiger',
                              'version'  :'Eiger_3_Apples.py',
                              'cmd'      :'dscan dummy 0 1 1 0.1',
                              'motors'   :[['dummy',0.0,1.0,1]],
                              'exp_time' :0.1}
        
        #### up to here

        #### the active components dict can be modified next to include only the actually present components:
        
        if active_components == None:
            self.active_components = {}
            self.active_components.update({'calibration'         :calib_dict,
                                           'Eiger4M'             :eiger_dict,
                                           'xia'                 :xia_dict,
                                           'spec'                :spec_dict,
                                           'beamline_positioners':beamline_dict,
                                           'initial_command'     :command_dict,
                                           'sample_info'         :sample_dict})

        else:
            print 'TODO get these components form some configuration file'
            
        
    def create_file(self, fname = None):
        '''
        WARNING this overwrites self.fname
        '''
        if fname == None:
            fname = self.fname
        else:
            self.fname = fname
            
        if os.path.exists(fname):
            os.remove(fname)
        self.nx_f = nx_f = nx.nxload(fname, 'w')
        nx_f.attrs['file_name']        = fname
        nx_f.attrs['creator']          = 'nexus_class.py'
        nx_f.attrs['HDF5_Version']     = h5py.version.hdf5_version
        nx_f.attrs['NX_class']         = 'NXroot'
        nxt.timestamp(nx_f)
        
    def close_file(self):
        '''
        self.nx_f.close()
        flushes file to disk (AFAIK)
        '''
        ## Does not seem to impact the access to the file in self.nx_f = ?
        
        self.nx_f.close()
        self.nx_f = nx.nxload(self.nx_f.nxfilename)
        self.nx_f.close()

    def change_to_read_only(self):
        ''' 
        returns a read only version of self.nx_f
        '''
        self.close_file()
        return nx.nxload(self.fname, 'r')

    def save_initial(self):
        '''
        saves the initial status of the beamline at the start of a scan
        for the handling/ parsing see fileIO.hdf5.nexus_update_functions.update_group_from_file
        '''        
        nxentry = self.nx_f['entry'] = nx.NXentry()

        ## setup instrument
    
        nxinstrument = nxentry['instrument'] = nx.NXinstrument()
        nxinstrument.attrs['name']           = self.instrument_name
    
        ## setup sample
        
        nxsample = nxentry['sample']   = nx.NXsample()
        nxsample.attrs['prefix']       = self.prefix

        for (group, properties) in self.active_components.items():
            nxentry = nuf.update_group_from_file(self.nx_f['entry'], group, properties)


        self.set_default_links()
        print self.nx_f.tree
        self.close_file()            
        nxt.timestamp(self.nx_f)

        
    def set_default_links(self, default_detector = 'Eiger4M', verbose = True):
        '''
        links entry/data/data to the default detector, if possible also with scanned motors
        '''
        local_nxpath   = 'entry/data'
        
        self._link_detector(default_detector = default_detector,
                            local_nxpath = local_nxpath,
                            verbose = verbose)

        if default_detector == 'Eiger4M':
            get_axes = False # cant assing attributes to the linked Eiger data.
        else:
            get_axes = True  # to be tested on a case by case way. I see problems with Energy dependent data, but this may be very nice to have for ROIs.

        if get_axes:
            self._link_axes(local_nxpath = local_nxpath,
                            verbose = verbose)
            self.nx_f[local_nxpath].attrs['sigal'] = 'data'
        
        print 'linked default dataset:'
        print self.nx_f['entry/data'].tree

    def _link_detector(self,
                       local_nxpath='entry/data',
                       default_detector = 'Eiger4M',
                       verbose = True):
            
        if verbose:
            print 'getting default detector path'

        data_nxpath = self.get_detector_path(default_detector)

        if verbose:
            print 'got ', data_nxpath
        
        self.nx_f['entry/data'] = self.nx_f[data_nxpath]
        
               
        
    def _link_axes(self,
                   local_nxpath   = 'entry/data',
                   verbose = True):
        '''
        only marginally tested
        '''
        if verbose:
            print 'getting scanned axes links'  
        axesnames = self.get_scanned_axesnames()
        
        axes = []
        for i, axisname in enumerate(axesnames):
            axis_path = self.get_scannedmotor_path(axisname)
            axes.append(axis_path)
            self.nx_f[local_nxpath][axisname] = self.nx_f[axis_path]
            
        self.nx_f[local_nxpath].attrs['axes'] = axesnames
        
        if verbose:
            print 'found and linked : ', axes



        
    def get_scanned_axesnames(self):
        '''
        returns list with the names of the scanned axes
        '''
        axes = self.nx_f['entry/instrument/initial_command/'].axes.split(':')
        return axes
        
    def get_detector_path(self, detector_name):
        '''
        returns the data of <detector_name>
        '''
        detectorpath = 'entry/instrument/'+ detector_name + '/data'
        return detectorpath

    def get_detector(self, motor_name):
        return self.nx_f[self.get_detector_path(motor_name)]
    
    def get_scannedmotor_path(self, motor_name):
        '''
        returns the path to the best guess of where the latest motor positions may be
        '''
        blp = self.nx_f['entry/instrument/beamline_positioners']

        if self.scantype.find('premaster'):
            scannedmotor_path = 'entry/instrument/initial_command/' + motor_name
        else:
            blp_final = blp['positioners_final']
            scannedmotor_path = nxt.find_dataset_path(blp_final, motor_name)                            
        return scannedmotor_path

    def get_scannedmotor(self, motor_name):
        return self.nx_f[self.get_scannedmotor_path(motor_name)]
    
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
        
        
    def do_it(self):
        '''
        just shorthand for development debugging
        '''
        self.create_file()
        self.save_initial()



        
class nx_id13_eh2_premaster(nx_id13):
    '''
    parent: nx_id13
    This class handles the writing of the eh2 data master file
    '''
    scantype = 'eh2_premaster' 

    def __init__(self,
                 fname = "/data/id13/inhouse6/COMMON_DEVELOP/py_andreas/nexustest_aj/pre_master.h5",
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
    scantype = 'eh3_premaster'

    def __init__(self,
                 fname = "/data/id13/inhouse6/COMMON_DEVELOP/py_andreas/nexustest_aj/pre_master.h5",
                 active_components=None,
                 intial = True):

        ## each of these nx instances is fixed to one file
        self.fname      = fname
        self.prefix     = 'TODO parse from fname'
        self.instrument_name = 'id13_eh3'
        self.update_components(active_components)
        

