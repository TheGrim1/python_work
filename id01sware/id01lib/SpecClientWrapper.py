# 20170611
# SL - Library of functions for interaction with spec variables with python
#
# to be integrated into existing scripts - *_generic.py
#
# BEWARE: generally this is dangerous as it is run in the background without specs knowledge
# 
# 
################### le code #################

try:
    from SpecClient_gevent import SpecVariable
    from SpecClient_gevent import SpecCommand
except ImportError:
    from SpecClient import SpecVariable
    from SpecClient import SpecCommand

import numpy as np


class SpecClientSession(object):
    def __init__(self, sv_limaroi = 'LIMA_ROI',
                       sv_limadev = 'LIMA_DEV',
                       specname ='nano2:psic_nano',
                       verbose=True):
        self.sv_limaroi = sv_limaroi
        self.sv_limadev = sv_limadev
        self.specname = specname
        self.device=''
        self.varcache = dict()
        self.speccmd = SpecCommand.SpecCommand('',self.specname)
        self.verbose = verbose
        
    def get_sv(self, sv):
        if sv not in self.varcache:
            _sv = SpecVariable.SpecVariable(sv,self.specname)
            self.varcache[sv] = _sv
        else:
            _sv = self.varcache[sv]
        if self.verbose:
            print('polling %s'%sv)
        return _sv.getValue()
        
    def send_sc(self,sc):
        self.speccmd.executeCommand(sc)
        
    def set_sv(self, sv, sv_value):
        self.send_sc(sv+'='+str(sv_value))
        return self.get_sv(sv)

    def find_roi_list(self):
        '''
        Finds first active detector from list and populates a ROI_LIST
        '''
        # limaroi params
        _limaroi = self.get_sv(self.sv_limaroi) # TODO: Check if PSCAN_ROICOUNTER[] is better
        no_rois = int(_limaroi['0'])

        # limadevices params
        _limadev =  self.get_sv(self.sv_limadev)
        no_devs = int(_limadev['0'])

        # find rois for active detector
        self.devices = devices = []
        for ii in range(1,no_devs+1,1):
            #print _limadev['%i'%ii]
            name = _limadev['%i'%ii]
            dev = _limadev[name]
            if dev.get('active', False) == '1':
                devices.append(name)
        
        self.device = devices[0] # just takes the first active camera
        roi_list=[]
        for i in range(1,no_rois+1,1):
            name = _limaroi['%i'%i]
            roidict = _limaroi[name]
            if roidict['ccdname']=="%s"%self.device:
                roi_list.append(name)
        
        return roi_list, self.device


    def get_last_image(self, device):
        '''
            get last image from the detector
        '''
        _limadev =  self.get_sv(self.sv_limadev)
        return self.get_sv('image_data%i'%int(_limadev["%s"%device]["unit"]))
        
    def get_pscan_vars(self,sv="PSCAN_ARR",pscan_live=False):
        '''
        extract pscan params
        '''
        _pscan_vars = self.get_sv(sv)

        # get motor names
        _m1_nm = _pscan_vars["header/cmd"].split()[1]
        _m2_nm = _pscan_vars["header/cmd"].split()[5]

        # motor start&end positions
        _m1_se = map(float,_pscan_vars["header/cmd"].split()[2:4])
        _m2_se = map(float,_pscan_vars["header/cmd"].split()[6:8])

        # get kmap column number
        piezo_counters = {'piy':'adcX','pix':'adcY','piz':'adcZ'}
        _m1 = _pscan_vars['header/cols'].split().index(piezo_counters[_m1_nm])
        _m2 = _pscan_vars['header/cols'].split().index(piezo_counters[_m2_nm])

        # find motor positions
        _m1_pos = self.get_sv('pscan_countersdata%i'%_m1)
        _m2_pos = self.get_sv('pscan_countersdata%i'%_m2)
        
        _m1_pts = int(_pscan_vars["header/cmd"].split()[4])
        _m2_pts = int(_pscan_vars["header/cmd"].split()[8])
        if pscan_live:
          return _m1_nm, _m1_se, _m1_pos, _m1_pts, _m2_nm, _m2_se, _m2_pos, _m2_pts,_pscan_vars
        else:
          return _m1_nm, _m1_se, _m1_pos, _m2_nm, _m2_se, _m2_pos, _pscan_vars
        




