# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:02:05 2017

@author: OPID13
"""
from __future__ import print_function
from __future__ import division


import sys, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import center_of_mass as com # only for EH3_hex_phikappa_gonio so far

## local imports
path_list = os.path.dirname(__file__).split(os.path.sep)
importpath_list = []
if 'skript' in path_list:
    for folder in path_list:
        importpath_list.append(folder)
        if folder == 'skript':
            break
importpath = os.path.sep.join(importpath_list)
sys.path.append(importpath)        


from fileIO.images.image_tools import optimize_greyscale
import fileIO.images.image_tools as it
from simplecalc.slicing import troi_to_slice

#import simplecalc.image_align as ia
#import simplecalc.fitting as fit
import fileIO.plots.plot_array as pa
import fileIO.datafiles.save_data as save_data
import fileIO.datafiles.open_data as open_data
from cameraIO.CamView_tools import stage
import cameraIO.lut_move as LUTs
import simplecalc.centering as cen


class phi_kappa_gonio(stage):
    def __init__(self, specsession = 'navitar', initialize_cameras = True):
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'top':
                               {'camera_index':0, 'horz_func':'x', 'vert_func':'y','focus':'z'},
                               'side':
                               {'camera_index':1, 'horz_func':'y', 'vert_func':'z','focus':'x'}})
                              
            self.initialize_cameras(plot=False,camera_type='usb')                        
            self.background = {}


            # General point of reference
            self.cross_pxl = {}
            self.cross_pxl['top'] = [600,1000]
            self.cross_pxl['side'] = [600,1000]
        self.reference_image = {}            
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'      : {'specname':'navix','is_rotation':False},
                      'y'      : {'specname':'naviy','is_rotation':False},
                      'z'      : {'specname':'naviz','is_rotation':False},
                      'phi'    : {'specname':'smsrotz','is_rotation':True},
                      'kappa'  : {'specname':'smsroty','is_rotation':True}}
        
        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}
        
        # lists of motors that will move the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>',<motor_parallel_in_view>],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not rigt handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'kappa':{'motors':['y','z','x'],'view':'side','invert':True}} 
        self.stagegeometry['COR_motors'].update({'phi':{'motors':['x','y','z'],'view':'top','invert':True}})

        # connectto spec
        self.connect(specsession = specsession)
        # initializing the default COR at the current motor positions
        self.COR = {}
        print(self.stagegeometry['COR_motors'])
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]

        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1) 
        
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['side'] = {}
        self.stagegeometry['same_calibration']['top']  = {}
        self.stagegeometry['same_calibration']['side'].update({'set1':{'x':1,'y':1,'z':1}})
        self.stagegeometry['same_calibration']['top'].update({'set1':{'x':1,'y':1,'z':1}})      
        
        self.calibration = {}
        self.calibration.update({'side':{}})
        self.calibration.update({'top':{}})
        print('setting default calibration for zoomed out microscopes')
        self._calibrate('y',-1495.4,'side')
        self._calibrate('y',1495.4,'top')
        self._calibrate('z',-914.02,'side')

        # lookuptables:
        self.lookup = LUTs.LUT_Navitar()
        
        # lookuptables look like this:
        # self.lookup[motor] = {} # look up dict for <motor>
        ## IMPORTANT, all defined lookup positions are referenced to the SAME positions for <motor>, here <kappas>
        ## positions in otor must be sorted! 
        # eg. and empty <kappa> lookup for <mot0> and <mot1>:
        # kappas = np.arange(360)
        # shift = np.zeros(shape = (kappas.shape[0],2))
        # self.lookup[motor].update({motor: kappas})  
        # self.lookup[motor].update({mot0: shift[:,0]*self.calibration[view][mot0]})
        # self.lookup[motor].update({mot1: shift[:,1]*self.calibration[view][mot1]})

        ## else we assume that all motors are correctly defined in the self.lookup[motor] dict!
        
    def connect_phi_to_kappa(self):
        '''
        after having loaded or made a lookuptable for function 'phi', this will change the type of self.lookup['phi'] to the UserDict class 'lookupdict_phi_kappa'.
        now the lookuptable is dependent on the current values of the function 'kappa'.
        this may disturbt the correct funcitonlaity of the inherited 'make_lookup' function of the stage class (TODO)
        '''
        self.lookup.link_dynamic()

        
        
    def do_gonio_docu(self, phi_pos=[x*25.0 for x in range(int(725/25))], kappa_pos=[x*15.0-45 for x in range(int(90/15))]):
        
        print('prepping images... ')
        prep_image = self._get_view('side')     

        shape = tuple([int(x) for x in [len(phi_pos)]+list(prep_image.shape)])
        
        for kap in kappa_pos:
            topstack= np.zeros(shape=shape)
            sidestack= np.zeros(shape=shape)
            self.mv('kappa',kap,move_using_lookup=True)

            print('doing backlashcorrection')
            
            self.mv('phi', phi_pos[0],move_using_lookup=True)
            self._backlash('phi',5.0)


            for i, pos in enumerate(phi_pos):

                title = 'frame %s of %s at pos = %s, kappa = %s'%(i+1, len(phi_pos), pos, kap)
                print(title)
                self.mv('phi', pos, move_using_lookup=True)
                topstack[i] = self._get_view('top')
                sidestack[i] = self._get_view('side')
                

            print('returning phi')
            self.mv('phi', phi_pos[0], move_using_lookup=True)

            top_prefix = 'topview_kappa%s_phi_' %int(kap)
            side_prefix = 'sideview_kappa%s_phi_' %int(kap)
            import fileIO.images.image_tools as it
            it.save_series(topstack,savename_list=[top_prefix+ str(int(x)) + '.png' for x in phi_pos])
            it.save_series(sidestack,savename_list=[side_prefix+ str(int(x)) + '.png' for x in phi_pos])
            it.array_to_imagefile(topstack.sum(0),imagefname=top_prefix+"_sum.png")
            it.array_to_imagefile(sidestack.sum(0),imagefname=side_prefix+"_sum.png")


            
class EH2_phi_kappa_gonio(stage):
    '''
    TODO: include new lookuptable class
    '''
    def __init__(self, spechost = 'id13CTRL', specsession = 'zap', initialize_cameras = True):
        # def __init__(self, specsession = 'motexplore', initialize_cameras = True):

        if initialize_cameras:
            self.background = {}
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'up':
                               {'camera_index':0, 'horz_func':'x', 'vert_func':'z','focus':'y'},
                               'side':
                               {'camera_index':1, 'horz_func':'y', 'vert_func':'z','focus':'x'}})


            # General point of reference eg. beam pos
            self.cross_pxl = {}
            self.cross_pxl['up'] = [198,386]
            self.cross_pxl['side'] = [300,400]
            
            self.initialize_cameras(plot=False,camera_type='eth',cameralist = ['id13/limaccds/eh2-vlm1','id13/limaccds/eh2-vlm2'])
        self.reference_image = {}        
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'      : {'specname':'strx','is_rotation':False},
                      'y'      : {'specname':'stry','is_rotation':False},
                      'z'      : {'specname':'strz','is_rotation':False},
                      'phi'    : {'specname':'smsrotz','is_rotation':True},
                      'kappa'  : {'specname':'smsroty','is_rotation':False}}
        
        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}
        
        # lists of motors that will the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>',<motor_parallel_in_view>],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not rigt handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'kappa':{'motors':['x','z'],'view':'side','invert':True}} 
        self.stagegeometry['COR_motors'].update({'phi':{'motors':['x','y'],'view':'up','invert':True}})

        # connect to spec
        self.connect(specsession = specsession,spechost=spechost)
        # initializing the default COR at the current motor positions
        self.COR = {}
        print(self.stagegeometry['COR_motors'])
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]

        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1)
        
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['up'] = {}
        self.stagegeometry['same_calibration']['side']  = {}
        self.stagegeometry['same_calibration']['side'].update({'set1':{'x':1,'y':1,'z':1}})
        self.stagegeometry['same_calibration']['up'].update({'set1':{'x':1,'y':1,'z':1}})      
        
        self.calibration = {}
        self.calibration.update({'side':{}})
        self.calibration.update({'up':{}})
        print('setting default calibration for zoomed out microscopes')
        self._calibrate('y',-1763.4485047350154,'side')
        self._calibrate('x',-3900.6735342539277,'up')
        #self._calibrate('x',-387.6735342539277,'up')
        
        # lookuptables:
        self.lookup = {}
        self.tmp_lookup = {}
        self.saved_positions = {}
        self.lookup['kappa']={'kappa':[],'x':[],'z':[]}
        self.lookup['phi']={'phi':[],'x':[],'y':[]}
                    
        # lookuptables look like this:
        # self.lookup[motor] = {} # look up dict for <motor>
        ## IMPORTANT, all defined lookup positions are referenced to the SAME positions for <motor>, here <kappas>
        ## positions in otor must be sorted! 
        # eg. and empty <kappa> lookup for <mot0> and <mot1>:
        # kappas = np.arange(360)
        # shift = np.zeros(shape = (kappas.shape[0],2))
        # self.lookup[motor].update({motor: kappas})  
        # self.lookup[motor].update({mot0: shift[:,0]*self.calibration[view][mot0]})
        # self.lookup[motor].update({mot1: shift[:,1]*self.calibration[view][mot1]})

        ## else we assume that all motors are correctly defined in the self.lookup[motor] dict!
        
    def connect_phi_to_kappa(self):
        '''
        after having loaded or made a lookuptable for function 'phi', this will change the type of self.lookup['phi'] to the UserDict class 'lookupdict_phi_kappa'.
        now the lookuptable is dependent on the current values of the function 'kappa'.
        this may disturbt the correct funcitonlaity of the inherited 'make_lookup' function of the stage class (TODO)
        '''
        self.lookup.link_dynamic()
        

    def get_2_imagestacks(self,
                          motor,
                          positions,
                          troi = None,
                          cutcontrast = 0.5,
                          move_using_lookup = False,
                          backlashcorrection = True,
                          sleep=0):
        print('prepping images... ')
        upprep_image = self._get_view(view='up',troi=troi)
        sideprep_image = self._get_view(view='side',troi=troi)
             
        uptmp_file_fname = '/data/id13/inhouse8/tmp_upstack.tmp'
        sidetmp_file_fname = '/data/id13/inhouse8/tmp_sidestack.tmp'
        
        upshape = tuple([int(x) for x in [len(positions)]+list(upprep_image.shape)])
        sideshape = tuple([int(x) for x in [len(positions)]+list(sideprep_image.shape)])

        if np.asarray(upshape).prod() > 2e8:
            # aleviate memory bottlenecks
            print(('created tmp file: ',uptmp_file_fname))
            print(('created tmp file :',sidetmp_file_fname))
            upstack = np.memmap(uptmp_file_fname, dtype=np.float16, mode='w+', shape=upshape)
            sidestack = np.memmap(sidetmp_file_fname, dtype=np.float16, mode='w+', shape=sideshape)
            
        else:
            upstack = np.zeros(shape = upshape)
            sidestack = np.zeros(shape = sideshape)  
        
        
        if backlashcorrection:
            print('doing backlashcorrection')
            self.mv(motor, positions[0],move_using_lookup=move_using_lookup)
            self._backlash(motor,backlashcorrection)
                
        print('starting rotation...')
        for i, pos in enumerate(positions):

            title = 'frame %s of %s at pos = %s'%(i+1, len(positions), pos)
            print(title)
            self.mv(motor, pos,move_using_lookup=move_using_lookup,sleep=sleep)
            upstack[i] = self._get_view(view='up')
            sidestack[i] = self._get_view(view='side')
        
        cutcontrast = 0.5
        upstack=np.where(upstack<abs(cutcontrast)*np.max(upstack),0,upstack)
        sidestack=np.where(sidestack>abs(cutcontrast)*np.max(sidestack),0,sidestack)
        perc_low = 0.1
        perc_high = 99.9
        upstack = optimize_greyscale(upstack, perc_low=perc_low, perc_high=perc_high)

        return upstack, sidestack

    def make_lookup_2(self,
                      motor = 'phi',
                      viewlist = ['up','side'],
                      positions = [0,1,2,3,4,5],
                      mode = 'com',
                      lookup_motors = ['y','x'],
                      plot = True,
                      troi = None,
                      cutcontrast = 0.5,
                      cutpercentilelist=[90,90],
                      backlashcorrection = True,
                      savename = None,
                      move_using_lookup=False,
                      saveimages=False,
                      saveimages_prefix='lookup1',
                      sleep=0,
                      use_background=False,
                      return_imagestacks=False):
        
        upstack,sidestack = self.get_2_imagestacks(motor=motor,
                                                   positions=positions,
                                                   move_using_lookup=move_using_lookup,
                                                   cutcontrast=cutcontrast,
                                                   sleep=sleep)

        upcen = self.cross_pxl[viewlist[0]][1]
        upwidth = 5
        up = upstack.copy()
        up=np.asarray(up,dtype=np.int16)
        uplines = up[:,upcen-upwidth:upcen+upwidth,:]
        up=np.where(up>np.percentile(up,cutpercentilelist[0]),up,0)
        dummy, upCOR, upshift= cen.COR_from_sideview(uplines, thetas=positions, mode='com', return_shift=True)

        sidecen = self.cross_pxl[viewlist[1]][1]
        sidewidth = 5        
        side = sidestack.copy()
        side=np.asarray(side,dtype=np.int16)
        sidelines = side[:,sidecen-sidewidth:sidecen+sidewidth,:]
        side=np.where(side>np.percentile(side,cutpercentilelist[1]),side,0)
        dummy, sideCOR, sideshift= cen.COR_from_sideview(sidelines, thetas=positions, mode='com', return_shift=True)

        new_y = upshift/self.calibration[viewlist[0]][lookup_motors[0]]
        new_x = sideshift/self.calibration[viewlist[1]][lookup_motors[1]]
        
        mot0=lookup_motors[0]
        mot1=lookup_motors[1]
        shift_lookup={motor:positions,mot1:new_x,mot0:new_y}

        self.update_lookup(motor=motor,shift_lookup=shift_lookup,overwrite =(not move_using_lookup))
        
        if return_imagestacks:
            return shift_lookup, upstack, sidestack
        else:
            return shift_lookup

        
    def capture_background(self, view, plot=False, troi=None):
        if plot:
            self.background[view] = self.plot(view, troi=troi)
        else:
            self.background[view] = self._get_view(view, troi=troi)


    def dscan(self, motor, start, finish, intervals, exptime):
        
        command_list = ['dscan', self.motors[motor]['specname'], str(start), str(finish), str(intervals), str(exptime)]
        command = (' ').join(command_list)
        self.SpecCommand(command)

    def do_gonio_docu(self, phi_pos=[x*20.0 for x in range(19)], kappa_pos=[-30.0]):

        start     = -0.02
        finish    = -start
        intervals = 40
        exptime   = 0.1
        
        for kap in kappa_pos:

            self.mv('kappa',kap,move_using_lookup=True)
            for i, pos in enumerate(phi_pos):

                self.mv('phi', pos, move_using_lookup=True)

                self.dscan('y',start, finish, intervals, exptime)
                self.SpecCommand('dipcen')
                self.mvr('z',start)
                self.dscan('z',start,finish, intervals, exptime)
                self.mvr('z',finish)
                
                self.mv('phi', phi_pos[0], move_using_lookup=True)
               

            
        
class EH2_cameras(stage):
    '''
    no motors, can be used to test camera interface
    '''
#    def __init__(self, spechost = 'id13ctrl', specsession = 'zap', initialize_cameras = True):
    def __init__(self, specsession = 'motexplore', initialize_cameras = True):
        # General point of reference
        self.cross_pxl = {}
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            self.views = {}
            self.views.update({'top':
                               {'camera_index':0, 'horz_func':None, 'vert_func':None},
                               'side':
                               {'camera_index':1, 'horz_func':None, 'vert_func':None}})
            self.cross_pxl['up'] = [372,228]
            self.cross_pxl['side'] = [400,300]
            self.initialize_cameras(plot=False,camera_type='eth',cameralist = ['id13/limaccds/eh2-vlm1','id13/limaccds/eh2-vlm2'])


       

class motexplore_jul17(stage):
    '''
    updated nov17
    '''
    def __init__(self, spechost = 'lid13lab1', specsession = 'motexplore', initialize_cameras = True):
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'top':
                               {'camera_index':0, 'horz_func':'x', 'vert_func':'y','focus':'z'},
                               'side':
                               {'camera_index':1, 'horz_func':'y', 'vert_func':'z','focus':'x'}})
            
            # General point of reference
            self.cross_pxl = {}
            self.cross_pxl['top'] = [600,1000]
            self.cross_pxl['side'] = [600,1000]
            self.initialize_cameras(plot=False,camera_type='usb')
        self.reference_image = {}
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'      : {'specname':'navix','is_rotation':False},
                      'y'      : {'specname':'naviy','is_rotation':False},
                      'z'      : {'specname':'naviz','is_rotation':False},
                      'rotz'   : {'specname':'srotz','is_rotation':True}}

        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}

        # lists of motors that will the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>',<motor_parallel_in_view>],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not right handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'rotz':{'motors':['x','y','z'],'view':'top','invert':False}} 
    

        # connect to spec
        self.connect(specsession = specsession)

        # initializing the default COR at the current motor positions
        self.COR = {}
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]
        
        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1)
            
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['side'] = {'set1':{'x':1,'y':1}}
        self.stagegeometry['same_calibration']['top']  = {'set1':{'x':1,'y':1}}

        
        self.calibration = {}
        self.calibration.update({'side':{}})
        self.calibration.update({'top':{}})
        print('setting default calibration for zoomed out microscopes')
        self._calibrate('y',-1495.4,'side')
        self._calibrate('y',1495.4,'top')
        self._calibrate('z',-914.02,'side')
        # lookuptables:
        self.lookup = LUTs.LUT_Generic(self.motors,self.stagegeometry)
        
class EH3_cameras_apr18(stage):
    '''
    updated apr18
    '''
    def __init__(self, spechost= 'lid13eh31', specsession = 'eh3', initialize_cameras = True):
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'vlm1':
                               {'camera_index':0, 'horz_func':'y', 'vert_func':'z','focus':'x'},
                               'vlm2':
                               {'camera_index':1, 'horz_func':'y', 'vert_func':'z','focus':'x'}})
            
            # General point of reference
            self.cross_pxl = {}
            self.cross_pxl['vlm1'] = [200,300]
            self.cross_pxl['vlm2'] = [200,300]
            self.initialize_cameras(plot=False,camera_type='eth',cameralist = ['id13/limaccds/eh3-vlm1','id13/limaccds/eh3-vlm2'])
        self.reference_image = {}
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'      : {'specname':'nnp1','is_rotation':False},
                      'y'      : {'specname':'nnp2','is_rotation':False},
                      'z'      : {'specname':'nnp3','is_rotation':False},
                      'rotz'   : {'specname':'dummy0','is_rotation':True}}

        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}

        # lists of motors that will the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>',<motor_parallel_in_view>],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not right handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'rotz':{'motors':['x','y'],'view':None,'invert':False}} 
    

        # connect to spec
        self.connect(spechost=spechost,specsession = specsession)

        # initializing the default COR at the current motor positions
        self.COR = {}
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]
        
        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1)
            
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['vlm1'] = {'set1':{'x':1,'y':1}}
        
        self.calibration = {}
        self.calibration.update({'vlm1':{}})
        print('setting default calibration for zoomed out microscopes')
        self._calibrate('y',-1495.4,'vlm1')
        self._calibrate('z',-914.02,'vlm1')
        # lookuptables:
        self.lookup = LUTs.LUT_Generic(self.motors,self.stagegeometry)


        
class EH3_smrhex_mai18(stage):
    '''
    updated mai 18
    '''
    def __init__(self, spechost = 'lid13eh31', specsession = 'eh3', initialize_cameras = True):
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'vlm1':
                               {'camera_index':0, 'horz_func':'y', 'vert_func':'z','focus':'x'},
                               'vlm2':
                               {'camera_index':1, 'horz_func':'y', 'vert_func':'z','focus':'x'}})
            
            # General point of reference
            self.cross_pxl = {}
            self.cross_pxl['vlm1'] = [576/2,748/2]
            self.cross_pxl['vlm2'] = [576/2,748/2]
            self.initialize_cameras(plot=False,camera_type='eth',cameralist = ['id13/limaccds/eh3-vlm1','id13/limaccds/eh3-vlm2'])
        self.reference_image = {}
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'      : {'specname':'nnx', 'is_rotation':False},
                      'y'      : {'specname':'nny', 'is_rotation':False},
                      'z'      : {'specname':'nnz', 'is_rotation':False},
                      'rotz'   : {'specname':'smphi', 'is_rotation':True}}

        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}

        # lists of motors that will the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>'],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not right handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'rotz':{'motors':['x','y'],'view':'None','invert':False}} 
    

        # connect to spec
        self.connect(spechost=spechost, specsession=specsession)

        # initializing the default COR at the current motor positions
        self.COR = {}
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]
        
        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1)
            
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['vlm1'] = {'set1':{'x':1,'y':-1,'z':1}}

        
        self.calibration = {}
        self.calibration.update({'vlm1':{}})
        print('setting default calibration for 5x microscopes')
        self._calibrate('y',-394.830502052507,'vlm1')
        # lookuptables:
        self.lookup = LUTs.LUT_Generic(self.motors,self.stagegeometry)
        
class EH3_XYTHetahex_mai18(stage):
    '''
    updated mai 18
    '''
    def __init__(self, spechost = 'lid13eh31', specsession = 'eh3', initialize_cameras = True):
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'vlm1':
                               {'camera_index':0, 'horz_func':'Y', 'vert_func':'hex_z','focus':'hex_x'},
                               'vlm2':
                               {'camera_index':1, 'horz_func':'y', 'vert_func':'z','focus':'x'}})
            
            # General point of reference
            self.cross_pxl = {}
            self.cross_pxl['vlm1'] = [576/2,748/2]
            self.cross_pxl['vlm2'] = [576/2,748/2]
            self.initialize_cameras(plot=False,camera_type='eth',cameralist = ['id13/limaccds/eh3-vlm1','id13/limaccds/eh3-vlm2'])
        self.reference_image = {}
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'hex_x'      : {'specname':'nnx', 'is_rotation':False},
                      'hex_y'      : {'specname':'nny', 'is_rotation':False},
                      'hex_z'      : {'specname':'nnz', 'is_rotation':False},
                      'X'   : {'specname':'X', 'is_rotation':False},
                      'Y'   : {'specname':'Y', 'is_rotation':False},
                      'turret'   : {'specname':'turret', 'is_rotation':False}, # todo
                      'Theta'   : {'specname':'Theta', 'is_rotation':True}}

        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}

        # lists of motors that will the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>'],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not right handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'Theta':{'motors':['Y','hex_z','hex_y'],'view':None,'invert':False}} 
    

        # connect to spec
        self.connect(spechost=spechost, specsession=specsession)

        # initializing the default COR at the current motor positions
        self.COR = {}
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]
        
        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1)
            
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['vlm1'].update({'set2':{'hex_x':1,'hex_y':-1,'hex_z':1,'Y':1,'X':1}})

        self.calibration = {}
        self.calibration.update({'vlm1':{}})
        
        print('setting default calibration for 5X microscope')
        self._calibrate('hex_z',394.830502052507,'vlm1')

        # lookuptables:
        self.lookup = LUTs.LUT_Generic(self.motors,self.stagegeometry)

    def get_zoom(self):
        if self.wm('turret') == 999:
            self.zoom = 5
        elif self.wm('turret') == 111:
            self.zoom = 50
        else:
            self.zoom = 0
        return self.zoom

        
class EH3_smrhexpiezo_mai18(stage):
    '''
    updated mai 18
    '''
    def __init__(self, spechost = 'lid13eh31', specsession = 'eh3', initialize_cameras = True):
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'vlm1':
                               {'camera_index':0, 'horz_func':'y', 'vert_func':'z','focus':'x'},
                               'vlm2':
                               {'camera_index':1, 'horz_func':'y', 'vert_func':'z','focus':'x'}})
            
            # General point of reference
            self.cross_pxl = {}
            self.cross_pxl['vlm1'] = [576/2,748/2]
            self.cross_pxl['vlm2'] = [576/2,748/2]
            self.initialize_cameras(plot=False,camera_type='eth',cameralist = ['id13/limaccds/eh3-vlm1','id13/limaccds/eh3-vlm2'])
        self.reference_image = {}
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'      : {'specname':'nnp1', 'is_rotation':False},
                      'y'      : {'specname':'nnp2', 'is_rotation':False},
                      'z'      : {'specname':'nnp3', 'is_rotation':False},
                      'out_x'   : {'specname':'nnx', 'is_rotation':False},
                      'out_y'   : {'specname':'nny', 'is_rotation':False},
                      'out_z'   : {'specname':'nnz', 'is_rotation':False},
                      'rotz'   : {'specname':'smphi', 'is_rotation':True}}

        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}

        # lists of motors that will the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>'],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not right handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'rotz':{'motors':['y','z','x'],'view':None,'invert':False}} 
    

        # connect to spec
        self.connect(spechost=spechost, specsession=specsession)

        # initializing the default COR at the current motor positions
        self.COR = {}
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]
        
        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1)
            
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['vlm1'] = {'set1':{'x':1,'y':1,'z':1}}
        self.stagegeometry['same_calibration']['vlm1'].update({'set2':{'out_x':1,'out_y':-1,'out_z':1}})

        
        self.calibration = {}
        self.calibration.update({'vlm1':{}})
        print('setting default calibration for 5X microscope')
        self._calibrate('y',394830.502052507,'vlm1')
        self._calibrate('out_z',394.830502052507,'vlm1')

        # lookuptables:
        self.lookup = LUTs.LUT_Generic(self.motors,self.stagegeometry)
        

class lab_TOMO_navi_sep18(stage):
    '''
    updated sep 06
    '''
    def __init__(self, spechost = 'lid13lab1', specsession = 'navitar', initialize_cameras = True):
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'top':
                               {'camera_index':0, 'horz_func':'y', 'vert_func':'x','focus':'z'},
                               'side':
                               {'camera_index':1, 'horz_func':'x', 'vert_func':'z','focus':'y'}})
        
            # General point of reference
            self.cross_pxl = {}
            self.cross_pxl['top'] = [200,320]
            self.cross_pxl['side'] = [200,320]

            self.initialize_cameras(plot=False,camera_type='usb')
        self.reference_image = {}
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'      : {'specname':'navix','is_rotation':False},
                      'y'      : {'specname':'naviy','is_rotation':False},
                      'z'      : {'specname':'naviz','is_rotation':False},
                      'phi'    : {'specname':'smphi','is_rotation':True},
                      'kappa'  : {'specname':'smkappa','is_rotation':True}}

        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}

        # lists of motors that will the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>'],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not right handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'phi':{'motors':['y','x','z'],'view':'top','invert':False},
                                            'kappa': {'motors':['x','z','y'],'view':'side','invert':True}}                
    

        # connect to spec
        self.connect(spechost=spechost, specsession=specsession)

        # initializing the default COR at the current motor positions
        self.COR = {}
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]
        
        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1)
            
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['top'] = {'set1':{'x':1,'y':1,'z':1}}
        self.stagegeometry['same_calibration']['side'] = {'set1':{'x':1,'y':1,'z':1}}

        self.calibration = {}
        self.calibration.update({'side':{}})
        self.calibration.update({'top':{}})
        print('setting default calibration for zoomed out microscopes')
        self._calibrate('y',495.4,'top')
        self._calibrate('z',-497.828,'side')
        
        # lookuptables:
        self.lookup = LUTs.LUT_TOMO_Navitar()

class EH2_TOMO_navi_sep18(stage):
    '''
    updated sep 06
    '''
    def __init__(self, spechost = 'lid13eh21', specsession = 'scanning', initialize_cameras = True):
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'inline':
                               {'camera_index':0, 'horz_func':'y', 'vert_func':'z','focus':'x'},
                               'side':
                               {'camera_index':1, 'horz_func':'x', 'vert_func':'z','focus':'y'}})
        
            # General point of reference
            self.cross_pxl = {}
            self.cross_pxl['inline'] = [229,373]
            self.cross_pxl['side'] = [229,373]
            self.median_filter = 3

            self.initialize_cameras(plot=False,camera_type='eth',cameralist = ['id13/limaccds/eh2-vlm1','id13/limaccds/eh2-vlm2'])
        self.reference_image = {}
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'      : {'specname':'strx','is_rotation':False},
                      'y'      : {'specname':'stry','is_rotation':False},
                      'z'      : {'specname':'strz','is_rotation':False},
                      'phi'    : {'specname':'smphi','is_rotation':True},
                      'kappa'  : {'specname':'smkappa','is_rotation':True}}

        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}

        # lists of motors that will the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>'],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not right handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'phi':{'motors':['y','x','z'],'view':None,'invert':False},
                                            'kappa': {'motors':['x','z','y'],'view':'side','invert':True}}                
        # connect to spec
        self.connect(spechost=spechost, specsession=specsession)

        # initializing the default COR at the current motor positions
        self.COR = {}
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]
        
        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1)
            
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['inline'] = {'set1':{'x':1,'y':1,'z':1}}
        self.stagegeometry['same_calibration']['side'] = {'set1':{'x':1,'y':1,'z':1}}

        self.calibration = {}
        self.calibration.update({'side':{}})
        self.calibration.update({'inline':{}})
        print('setting default calibration for zoomed in microscope')
        self._calibrate('y',495.4,'inline')
        self._calibrate('z',-3424,'side')
        
        # lookuptables:
        self.lookup = LUTs.LUT_TOMO_Navitar()

    def make_tmp_lookup_both_views(self,
                                   motor = 'phi',
                                   views = ['inline','side'],
                                   positions = [0,1,2,3,4,5],
                                   mode = 'com',
                                   resolution = None,
                                   lookup_motors = ['y','x'],
                                   correct_vertical=True,
                                   plot = False,
                                   troi = None,
                                   cutcontrasts=[0.1,0.1],
                                   backlashcorrection = True,
                                   savename = None,
                                   move_using_lookup=False,
                                   saveimages=False,
                                   saveimages_prefix='lookup1',
                                   sleep=0,
                                   align_to_cross = True):
        ''' 
        creates a lookup table for <motor>
        corresponding movement command = self.mv(..., move_using_lookup = True,...)
        the lookuptable will contain positions of <motor> between 0 and 360 seperated by <resolution> 
        OR values for the defined list of angles <positions>
        for the motors listed under self.stagegeometry['COR_motors'][<motor>]['motors'] positions minimising the movement of the sample are found using the imagealigment mode 'mode' (see self.make_calibration)
        alternatively you can define motors <lookup_motors> [horz_motor_1, horz_motor_2, vert_motor_1]
        mode.upper() can be ['ELASTIX','COM','CC','USERCLICK','MASK_TL','MASK_TR','TOPMASK_l','TOPMASK_r']
        tries to align all positions with first position, which is kept as referemce_image
        if correct_vertical = False - ignores the vertical correction (sometimes good for needles)
        overwrites current lookup.tmp_lookup
        backlashcorrection = bool or value of correction
        '''

        if type(positions) == type(None):
            if type(resolution) == type(None):
                raise ValueError('please define either a <resolution> or a list <positions>')
            else:
                positions = [x*resolution for x in range(int(360/resolution))]

        if len(positions)>10:
            if plot==True:
                plot=False
                print('WARNING: too many positions, will not plot!')

        if lookup_motors == None: # assume the same motors as for COR
            mot0 = self.views[0]['horz_func']
            mot1 = self.views[1]['horz_func']
            mot2 = self.views[0]['vert_func']
        else:
            mot0 = lookup_motors[0]
            mot1 = lookup_motors[1]
            mot2 = lookup_motors[2]
        motor_list = [mot0,mot1,mot2]
            
        print('will try to get a lookuptable to align rotation in ', motor)
        print('with horizontal motor %' %(mot2))
        print('viewed from the ', views[1])
        print('and with motors %s (horz) an %s (vert)' %(mot0, mot1))
        print('viewed from the ', views[0])
        print('using alignment algorithm: ', mode)
                
        if plot > 1:
            plot_stack=True
        else:
            plot_stack=False
                
        if backlashcorrection:
            print('doing backlashcorrection')
            self.mv(motor,positions[0]+positions[0]-positions[1], move_using_lookup=move_using_lookup)
                
        self.lookup.initialize_tmp_lookup(lookupmotor=motor,save_motor_list=motor_list)

        print('\n\ngoing to first position of {}'.format(len(positions)))
        self.mv(motor,positions[0],move_using_lookup=move_using_lookup)
        if focus_motor_range != None:
            print('focussing')
            self.auto_focus(view=view,
                            motor=mot2,
                            motor_range=focus_motor_range,
                            plot=plot,
                            points=focus_points,
                            move_using_lookup=False,
                            troi=troi,
                            backlashcorrection=focus_motor_range/focus_points,
                            sleep=sleep)
        if sleep:
            print('sleeping {}s'.format(sleep))
            time.sleep(sleep)

        if not align_to_cross:
            self.update_reference_image(views[0])
            self.update_reference_image(views[1])
        else:
            if cutcontrasts[0]>0:
                self.reference_image[views[0]] = np.zeros_like(self._get_view(views[0]))
                self.reference_image[views[0]][self.cross_pxl[views[0]][0],self.cross_pxl[views[0]][1]] = 255
            else:
                self.reference_image[views[0]] = np.ones_like(self._get_view(views[0]))*255
                self.reference_image[views[0]][self.cross_pxl[views[0]][0],self.cross_pxl[views[0]][1]] = 0
            if cutcontrasts[1]>0:
                self.reference_image[views[1]] = np.zeros_like(self._get_view(views[1]))
                self.reference_image[views[1]][self.cross_pxl[views[1]][0],self.cross_pxl[views[1][1]]] = 255
            else:
                self.reference_image[views[1]] = np.ones_like(self._get_view(views[1]))*255
                self.reference_image[views[1]][self.cross_pxl[views[1]][0],self.cross_pxl[views[1]][1]] = 0
        print('got reference image')
        
        for i, pos in enumerate(positions[1:]):
            
            print('\n\ngoing to lookup position {} of {}'.format(i+2,len(positions)))            
            self.mv(motor,pos,move_using_lookup=move_using_lookup)

            
            print('aligning to reference image 1')
            self.align_to_reference_image(view=views[0],
                                          mode=mode,
                                          align_motors=motor_list, 
                                          correct_vertical=correct_vertical,
                                          focus_motor_range=None,
                                          refocus=refocus,
                                          plot=plot,
                                          troi=troi,
                                          cutcontrast=cutcontrasts[0],
                                          sleep=sleep)
            print('aligning to reference image 2')
            self.align_to_reference_image(view=views[1],
                                          mode=mode,
                                          align_motors=motor_list, 
                                          correct_vertical=False,
                                          focus_motor_range=None,
                                          refocus=refocus,
                                          plot=plot,
                                          troi=troi,
                                          cutcontrast=cutcontrasts[1],
                                          sleep=sleep)
            

                
            self.lookup.add_pos_to_tmp_lookup(motor,self._get_pos())
            if saveimages:
                save_image = self._get_view(view)
            if saveimages:
                image_fname = save_prefix+'{:6d}.png'.format(i)
                it.array_to_imagefile(save_image,image_fname)               

        print('\nDONE\nlookup ready to be saved in .tmp_lookup')

        
class lab_smrotgonio_navi_jul18(stage):
    '''
    updated jul 30
    '''
    def __init__(self, spechost = 'lid13lab1', specsession = 'navitar', initialize_cameras = True):
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'top':
                               {'camera_index':0, 'horz_func':'x', 'vert_func':'y','focus':'z'},
                               'side':
                               {'camera_index':1, 'horz_func':'y', 'vert_func':'z','focus':'x'}})
        
            # General point of reference
            self.cross_pxl = {}
            self.cross_pxl['top'] = [600,1000]
            self.cross_pxl['side'] = [600,1000]

            self.initialize_cameras(plot=False,camera_type='usb')
        self.reference_image = {}
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'      : {'specname':'navix','is_rotation':False},
                      'y'      : {'specname':'naviy','is_rotation':False},
                      'z'      : {'specname':'naviz','is_rotation':False},
                      'phi'    : {'specname':'smphi','is_rotation':True},
                      'kappa'  : {'specname':'smkappa','is_rotation':True}}

        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}

        # lists of motors that will the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>'],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not right handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'phi':{'motors':['x','y','z'],'view':None,'invert':False},
                                            'kappa': {'motors':['y','z','x'],'view':None,'invert':False}}                
    

        # connect to spec
        self.connect(spechost=spechost, specsession=specsession)

        # initializing the default COR at the current motor positions
        self.COR = {}
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]
        
        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1)
            
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['top'] = {'set1':{'x':1,'y':1,'z':1}}
        self.stagegeometry['same_calibration']['side'] = {'set1':{'x':1,'y':1,'z':1}}

        self.calibration = {}
        self.calibration.update({'side':{}})
        self.calibration.update({'top':{}})
        print('setting default calibration for zoomed out microscopes')
        self._calibrate('y',-1495.4,'side')
        self._calibrate('y',1495.4,'top')
        self._calibrate('z',-914.02,'side')
        
        # lookuptables:
        self.lookup = LUTs.LUT_Navitar()


        
class EH3_hex_phikappa_gonio(stage):
    def __init__(self, spechost = 'lid13eh31',  specsession = 'eh3', initialize_cameras = True):
        if initialize_cameras:
            # this list defines which camera is called by view, here view = 'top' -> camera 0:
            # and which motors will by default (cross_to function) move the sample in this view
            self.views = {}
            self.views.update({'sample':
                               {'camera_index':0, 'horz_func':'x', 'vert_func':'y','focus':'z'},
                               'wall':
                               {'camera_index':1, 'horz_func':'phi', 'vert_func':'kappa','focus':None}})
                              
            self.background = {}

            # General point of reference
            self.cross_pxl = {}
            self.cross_pxl['sample'] = (576/2, 748/2)
            self.cross_pxl['wall'] = (576/2, 748/2)
            self.initialize_cameras(plot=False,camera_type='eth',cameralist = ['id13/limaccds/eh3-vlm1','id13/limaccds/eh3-vlm2'])
        self.reference_image = {}
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'      : {'specname':'nnx','is_rotation':False},
                      'y'      : {'specname':'nny','is_rotation':False},
                      'z'      : {'specname':'nnz','is_rotation':False},
                      'phi'    : {'specname':'smphi','is_rotation':True},
                      'kappa'  : {'specname':'smkappa','is_rotation':True},
                      'Theta'  : {'specname':'Theta','is_rotation':True}}
        
        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeometry = {}
        
        # lists of motors that will move the rotation axis for centering
        # eg:
        # self.stagegeometry['COR_motors'] = {'<rotation_motor>':{['<motor_horz_in_view>','<motor_vert_in_view>',<motor_parallel_in_view>],
        #                                     'parallel_view':'<top/side>',
        #                                     'invert':<True/False>}} # invert if rotation not rigt handed with respect to the motors
        self.stagegeometry['COR_motors'] = {'kappa':{'motors':['y','z','x'],'view':'sample','invert':True}} 
        self.stagegeometry['COR_motors'] = {'phi':{'motors':['x','y','z'],'view':None,'invert':False}}
        self.stagegeometry['COR_motors'] = {'Theta':{'motors':['x','y','z'],'view':None,'invert':False}}

        # connectto spec
        self.connect(spechost = spechost, specsession = specsession)
        # initializing the default COR at the current motor positions
        self.COR = {}
        print(self.stagegeometry['COR_motors'])
        [self.COR.update({motor:[self.wm(COR_motor) for COR_motor in COR_dict['motors']]}) for motor,COR_dict in list(self.stagegeometry['COR_motors'].items())]

        # dicts of motors that can have the same calibration:
        # level 1 : which view (side or top)
        # level 2 : group of motors (any name, here 'set1'
        # level 3 : the motors with relative calibration factors (here 1) 
        
        self.stagegeometry['same_calibration'] = {}
        self.stagegeometry['same_calibration']['sample'] = {}
        self.stagegeometry['same_calibration']['wall']  = {}
        self.stagegeometry['same_calibration']['sample'].update({'set1':{'x':1,'y':1,'z':1}})
        self.stagegeometry['same_calibration']['wall'].update({'set1':{'phi':1,'kappa':1,'Theta':1}})      
        
        self.calibration = {}
        self.calibration.update({'wall':{}})
        self.calibration.update({'sample':{}})
        print('setting default calibration for zoomed out microscopes')
        self._calibrate('y',-1495.4,'sample')
        self._calibrate('phi',-537.8,'wall')

        # lookuptables:
        self.lookup = LUTs.LUT_Navitar()
        
        # lookuptables look like this:
        # self.lookup[motor] = {} # look up dict for <motor>
        ## IMPORTANT, all defined lookup positions are referenced to the SAME positions for <motor>, here <kappas>
        ## positions in otor must be sorted! 
        # eg. and empty <kappa> lookup for <mot0> and <mot1>:
        # kappas = np.arange(360)
        # shift = np.zeros(shape = (kappas.shape[0],2))
        # self.lookup[motor].update({motor: kappas})  
        # self.lookup[motor].update({mot0: shift[:,0]*self.calibration[view][mot0]})
        # self.lookup[motor].update({mot1: shift[:,1]*self.calibration[view][mot1]})

        ## else we assume that all motors are correctly defined in the self.lookup[motor] dict!

        ## addon for wall mounted camera:
        self.camera_position = {}
        self.home_camera_position

        
    def connect_phi_to_kappa(self):
        '''
        after having loaded or made a lookuptable for function 'phi', this will change the type of self.lookup['phi'] to the UserDict class 'lookupdict_phi_kappa'.
        now the lookuptable is dependent on the current values of the function 'kappa'.
        this may disturbt the correct funcitonlaity of the inherited 'make_lookup' function of the stage class (TODO)
        '''
        self.lookup.link_dynamic()

    def get_pos_from_wall(self):
        '''
        return the position of phi and kappa from the laserspot on the all as calibrated
        '''
        kappa_pxl, phi_pxl = self._get_camera_com()
        kappa_pos = (self.camera_position['kappa'] - kappa_pxl) / self.calibration['wall']['phi']
        phi_pos = (self.camera_position['phi'] - phi_pxl) /self.calibration['wall']['phi'] 
        
        return phi_pos, kappa_pos

    def _get_camera_com(self, cutcontrast=0.1,troi=[[2,2],[572, 744]],no_exposures=1):
        if no_exposures>0:
            image = np.asarray([self._get_view('wall',troi=troi) for i in range(no_exposures)]).sum(axis=0)
        else:
            image = self._get_view('wall',troi=troi)
        
        image = it.optimize_imagestack_contrast(imagestack=image,cutcontrast=cutcontrast)
        return com(image)        
        
    def home_camera_position(self):
        com = self._get_camera_com(no_exposures=10)
        calibration = self.calibration['wall']['phi']
        pos_dc = self.get_motor_dict()
        phi_real = pos_dc['phi']
        kappa_real = pos_dc['kappa']

        self.camera_position['kappa'] =  com[0] - kappa_real*calibration 
        self.camera_position['phi'] = com[1] - phi_real*calibration

        print('found kappa = {:.4f} at vert_pixel = {}'.format(kappa_real, com[0]))
        print('found phi   = {:.4f} at horz_pixel = {}'.format(phi_real, com[1]))
        
    def add_camera_positions(self, pos_dc):

        pos_phi, pos_kappa = self.get_pos_from_wall()
        pos_dc['phi_camera']=pos_phi
        pos_dc['kappa_camera']=pos_kappa
        return pos_dc

    
        
    def do_gonio_docu(self, phi_pos=[x*25.0 for x in range(int(725/25))], kappa_pos=[x*15.0-45 for x in range(int(90/15))]):

        ### TODO ### 
        print('prepping images... ')
        prep_image = self._get_view('side')     

        shape = tuple([int(x) for x in [len(phi_pos)]+list(prep_image.shape)])
        
        for kap in kappa_pos:
            topstack= np.zeros(shape=shape)
            sidestack= np.zeros(shape=shape)
            self.mv('kappa',kap,move_using_lookup=True)

            print('doing backlashcorrection')
            
            self.mv('phi', phi_pos[0],move_using_lookup=True)
            self._backlash('phi',5.0)


            for i, pos in enumerate(phi_pos):

                title = 'frame %s of %s at pos = %s, kappa = %s'%(i+1, len(phi_pos), pos, kap)
                print(title)
                self.mv('phi', pos, move_using_lookup=True)
                topstack[i] = self._get_view('top')
                sidestack[i] = self._get_view('side')
                

            print('returning phi')
            self.mv('phi', phi_pos[0], move_using_lookup=True)

            top_prefix = 'topview_kappa%s_phi_' %int(kap)
            side_prefix = 'sideview_kappa%s_phi_' %int(kap)
            import fileIO.images.image_tools as it
            it.save_series(topstack,savename_list=[top_prefix+ str(int(x)) + '.png' for x in phi_pos])
            it.save_series(sidestack,savename_list=[side_prefix+ str(int(x)) + '.png' for x in phi_pos])
            it.array_to_imagefile(topstack.sum(0),imagefname=top_prefix+"_sum.png")
            it.array_to_imagefile(sidestack.sum(0),imagefname=side_prefix+"_sum.png")
