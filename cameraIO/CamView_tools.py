# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:02:05 2017

@author: OPID13
"""


import sys, os
import matplotlib.pyplot as plt
import numpy as np
import time

# local imports
path_list = os.path.dirname(__file__).split(os.path.sep)
importpath_list = []
if 'skript' in path_list:
    for folder in path_list:
        importpath_list.append(folder)
        if folder == 'skript':
            break
importpath = os.path.sep.join(importpath_list)
sys.path.append(importpath)        

sys.path.insert(0, "/mntdirect/_data_opid13_inhouse/Manfred/PLATFORM/d_mnext3/mnext3/SW/muenchhausen/EIGER-DVP/.DVP_1000")



from fileIO.images.image_tools import optimize_greyscale
import fileIO.images.image_tools as it
from simplecalc.slicing import troi_to_slice
from SpecClient import SpecCommand, SpecMotor
# bla
import simplecalc.centering as cen
import simplecalc.image_align as ia
import simplecalc.fitting as fit
import fileIO.plots.plot_array as pa
import fileIO.datafiles.save_data as save_data
import fileIO.datafiles.open_data as open_data
from CamView_grabber import CamView_grabber

class stage():
    def __init__(self):
        # this list defines which camera is called by view, here view = 'top' -> camera 0:
        self.viewlist = ['top','side']
        self.saved_positions = {}
        

    def connect(self,spechost = 'lid13lab1', 
                 specsession = 'motexplore', 
                 timeout = 1000000000):
        print 'connecting to %s' % spechost
        self.spechost    = spechost
        print 'specsession named %s'  % specsession
        self.specsession = specsession
        self.timeout     = timeout
        self._specversion_update()
        self.COR = None
        
##### handling motors
    def _specversion_update(self):
        self.specversion = self.spechost + ':' + self.specsession
                   
    def _add_motors(self,**kwargs):
        self.motors.update(kwargs)
        for function, motdict in kwargs.items():
            if motdict['is_rotation']:
                rot_str = 'rotational'
            else:
                rot_str = 'linear'
            print 'added %s motor for %s%s called %s%s in this spec session' %(rot_str,(10-len(function))*' ',function,(10-len(motdict['specname']))*' ',motdict['specname'])

    def _correct_with_lookup(self, function, startpos_dc, end_pos):
        print 'correcting movement of %s with motors:' % function

        start_pos = startpos_dc[function]
        
        return_dict = {}
        if self.motors[function]['is_rotation']:
            # DEBUG
            print "this is a rotation"
            start_pos = start_pos % 360.0
            end_pos   = end_pos   % 360.0

        if function in self.lookup.lookup.keys():
            correction_dc = self.lookup.get_lut_correction(function, startpos_dc, end_pos, dynamic = True)
            return correction_dc
        else:
            print 'no lookuptable found for ' , function

    
    def mvr(self, function, distance,
            move_in_pxl = False, view = 'side',
            move_using_lookup = False, sleep=0):
        if move_in_pxl:
            distance = distance / self.calibration[view][function]

        startpos_dc = self._get_pos()

        cmd = SpecCommand.SpecCommand('mvr', self.specversion, self.timeout)
        print 'mvr %s %s' %(function, distance)
        cmd(self.motors[function]['specname'], distance)

        ### optional correction of motors using lookuptable
        if move_using_lookup:
            end_pos = self.wm(function)
            correct_dict = self._correct_with_lookup(function, startpos_dc, end_pos)
            for mot, correction in correct_dict.items():
                self.mvr(mot, correction)
        if sleep:
            time.sleep(sleep)        
        
    def mv(self, function, position,
           move_using_lookup = False,sleep=0):

        startpos_dc = self._get_pos() # needed for move_using_lookup
 
        cmd = SpecCommand.SpecCommand('mv', self.specversion, self.timeout)  
        print 'mv %s %s' %(function, position)
        cmd(self.motors[function]['specname'], position)
        
        ### optional correction of motors using lookuptable
        if move_using_lookup:
            end_pos = self.wm(function)
            correct_dict = self._correct_with_lookup(function, startpos_dc, end_pos)
            for mot, correction in correct_dict.items():
                self.mvr(mot, correction)
        if sleep:
            time.sleep(sleep)
                
    def cross_to(self, horz_pxl=None,
                 vert_pxl=None,
                 view = 'side',
                 move=True,
                 move_using_lookup=False):
        if vert_pxl==None:
            vert_pxl=self.cross_pxl[view][0]
        if horz_pxl==None:
            horz_pxl=self.cross_pxl[view][1]
            
        dhorz = (self.cross_pxl[view][1] - horz_pxl)
        dvert = (self.cross_pxl[view][0] - vert_pxl)
        
        if view == 'side':
            horz_func = 'x'
            vert_func = 'z'
        elif view == 'top':
            horz_func = 'x'
            vert_func = 'y'
        elif view == 'up':
            horz_func = 'x'
            vert_func = 'z'
        else:
            raise ValueError(view + ' is not a valid view!')

        if move:
            self.mvr(horz_func, dhorz, move_in_pxl = True, view = view, move_using_lookup=move_using_lookup)
            self.mvr(vert_func, dvert, move_in_pxl = True, view = view, move_using_lookup=move_using_lookup)
            return [dhorz, dvert]
        else:
            return [dhorz, dvert]
                             
    def wm(self, function):
        # print 'getting position of motor %s'% self.motors[function]['specname']
        specmotor = SpecMotor.SpecMotor(self.motors[function]['specname'], self.specversion)  
        return specmotor.getPosition()

    def _get_pos(self):
        pos_dc = {}
        for function in self.motors.keys():
            pos_dc.update({function: self.wm(function)})

        return pos_dc
            
        
    def _backlash(self, function, backlashcorrection, sleep=0):
        if type(backlashcorrection) == bool:
            distance = 5.0
        else:
            distance = backlashcorrection
        self.mvr(function, -1.0*distance)
        self.mvr(function, distance, sleep=sleep)


    def _calibrate(self, motor, calibration, view = 'side'):
        ''' itererate through the 'same calibration' dict of dicts
        to set all relative calibrations
        '''
        calibration=float(calibration)
        self.calibration[view][motor] = float(calibration)
        for motorset in self.stagegeometry['same_calibration'][view].values():
            if motor in motorset.keys():
                for othermotor in motorset.keys():
                    # this is 1 for the initial motor, and equal to the relative factor for all others
                    factor = motorset[othermotor]/motorset[motor] 
                    self.calibration[view].update({othermotor:factor*calibration}) 
        
    def SpecCommand(self, command):
        print 'sending %s the command:'% self.specversion
        print command
        cmd = SpecCommand.SpecCommand(command , self.specversion, self.timeout)  
        cmd()

    def save_position(self,position_name):
        save_dict = {}
        
        for mot in self.motors.keys():
            pos = self.wm(mot)
            save_dict.update({mot:pos})
            print('%s read pos: %s'% (mot,pos))
            self.saved_positions.update({position_name:save_dict})
        return save_dict

    def restore_position(self,position_name,silent=False,sleep=0):

        if not silent:
            awns = raw_input('if you want to go here press <y>')
            if awns=='y':
                for mot,pos in self.saved_positions[position_name].items():
                    self.mv(mot, pos)
            else:
                print('did nothing')
        else:
            for mot,pos in self.saved_positions[position_name].items():
                self.mv(mot, pos)
        if sleep:
            time.sleep(sleep)
                    

    
##### handling the cameras
    def initialize_cameras(self, **kwargs):
        '''
        kwargs:
        plot=False, 
        camera_type ='usb'
        cameralist = ['id13/limaccds/eh2-vlm1','id13/limaccds/eh2-vlm2']
        '''
        self.cameras =  CamView_grabber(**kwargs)

        if 'plot' in kwargs:
            if kwargs['plot']:
                for view in self.viewlist:
                    self.plot(view,title = view)
    
    def _get_view(self, view='top', troi = None):        
        
        cam_no = None
        for i,cam in enumerate(self.viewlist):
            if view.upper() == cam.upper():
                cam_no = i
        if type(cam_no)==type(None):
            print('could not find view %s in viewlist ' %view, self.viewlist)
        else:
            return self.cameras.grab_image(cam_no,troi)
    
 
        
    def plot(self, view = 'top', title = '', troi = None):
        '''
        view = 'top' or 'side'
        '''
        plt.ion()
        image0    = self._get_view(view,troi=troi)            
        fig0, ax0 = plt.subplots(1) 
        ax0.imshow(image0)
        ax0.plot(self.cross_pxl[view][1],self.cross_pxl[view][0],'rx')
        ax0.set_title(view + ' view ' + title)
        plt.show()
        plt.ioff()
        return image0
        
##### centering functionality:
    def _goto_COR(self, motor = 'rotz', move_using_lookup=False):
        COR_motors = self.stagegeometry['COR_motors'][motor]['motors']
        for i, COR_mot in enumerate(COR_motors):
            self.mv(COR_mot, self.COR[motor][i],move_using_lookup=move_using_lookup)
        return 1

    def center_COR(self, 
                   motor     = 'rotz',
                   view      = 'side', 
                   positions    = [x*180.0/10 for x in range(10)],
                   mode      = 'COM',
                   move_using_lookup = False,
                   plot      = False,
                   troi      = None,                   
                   cutcontrast = 0.5,                   
                   backlashcorrection = True):
        '''
        moves the COR to the position of the first image
        view   = 'top' or 'side'
        plot   = Boolean(True)
        troi   = ((top,left),(height,width)) or None
        positions = list of absolute angles to rotate stage to
        mode   = centering algorithm used : 'COM', 'CC', 'elastix'
        backlashcorrection = Bool(True)
        motor  = 'rotz' stage name for the rotation to be centered
        cutcontrast [-1.0:1.0](optional) use a negative value to cut dark values, positive for bright values
        '''
        self.get_COR(view      = view,
                     plot      = plot,
                     troi      = troi,
                     positions = positions,
                     mode      = mode,
                     motor     = motor,
                     move_using_lookup=move_using_lookup,
                     cutcontrast=cutcontrast,
                     backlashcorrection = backlashcorrection)
        
        if self._goto_COR(motor = motor):
            print 'SUCCESS, now move your sample into the focus and repeat until COR is sufficiently aligned.' 
        
    def _get_imagestack(self, view,
                        motor,
                        plot,
                        positions,
                        troi = None,
                        cutcontrast = 0.5,
                        move_using_lookup = False,
                        backlashcorrection = True,
                        temp_file_fname = 'tmp_imagestack.tmp',
                        sleep = 0):

        print 'prepping images... '
        prep_image = self._get_view(view,troi=troi)
             

        shape = tuple([int(x) for x in [len(positions)]+list(prep_image.shape)])
        if np.asarray(shape).prod() > 2e8:
            # aleviate memory bottlenecks
            print('created temp file: ',temp_file_fname) 
            imagestack = np.memmap(temp_file_fname, dtype=np.float16, mode='w+', shape=shape)
        else:
            imagestack = np.zeros(shape = shape)  
        
        
        if backlashcorrection:
            print 'doing backlashcorrection'
            self.mv(motor, positions[0],move_using_lookup=move_using_lookup)
            self._backlash(motor,backlashcorrection,sleep=sleep)
                
        print 'starting rotation...'
        for i, pos in enumerate(positions):

            title = 'frame %s of %s at pos = %s'%(i+1, len(positions), pos)
            print title
            self.mv(motor, pos,move_using_lookup=move_using_lookup,sleep=sleep)
            if plot:
                imagestack[i] = self.plot(view, title, troi = troi)
            else:
                imagestack[i] = self._get_view(view, troi)

        print 'returning %s' %motor
        self.mv(motor, positions[0],move_using_lookup=move_using_lookup)

        print 'optimizing image contrast'

        if cutcontrast > 0:
            print 'cutting low intensities'
            imagestack=np.where(imagestack<abs(cutcontrast)*np.max(imagestack),0,imagestack)
            perc_low = 1
            perc_high = 100
        else:
            print 'cutting high intensities'
            imagestack=np.where(imagestack>abs(cutcontrast)*np.max(imagestack),np.max(imagestack),imagestack)
            perc_low=0
            perc_high =99

        imagestack = optimize_greyscale(imagestack, perc_low=perc_low, perc_high=perc_high)


        return imagestack
            
    def get_COR(self, 
                motor     = 'rotz',
                view      = 'side', 
                positions    = [x*180.0/10 for x in range(10)],
                mode      = 'COM',
                plot      = False,
                troi      = None,
                cutcontrast = 0.5,
                move_using_lookup= False,
                backlashcorrection = True,
                saveimages = False,
                saveimages_prefix = '',
                sleep=0):
        '''
        view   = 'top' or 'side'
        plot   = Boolean(True)
        troi   = ((top,left),(height,width)) or None
        positions = list of absolute angles to rotate stage to
        mode   = centering algorithm used : 'COM', 'CC', 'elastix'
        backlashcorrection = Bool(True)
        motor  = 'rotz' stage name for the rotation to be centered
        cutcontrast [-1.0:1.0](optional) use a negative value to cut dark values, positive for bright values
        move_using_lookup [boolean]
        '''
        if mode.upper() not in ['ELASTIX','COM','CC']:
            raise NotImplementedError(mode ,' is not a valid image alignment mode for getting the COR')

        if plot > 1:
            plot_imagestack = True
        else:
            plot_imagestack = False
                
        imagestack = self._get_imagestack(view = view,
                                          plot = plot_imagestack,
                                          troi = troi,
                                          positions = positions,
                                          motor  = motor,
                                          cutcontrast = cutcontrast,
                                          move_using_lookup=move_using_lookup,
                                          backlashcorrection = backlashcorrection,
                                          sleep=sleep)
                                          
        print 'calculating COR'

        # if the rotation axis is inverted with respect to the motor definitions:
        # this stays hidden to the user
        calc_positions = np.copy(positions)
        if self.stagegeometry['COR_motors'][motor]['invert']:
            calc_positions *= -1
        
        if view == self.stagegeometry['COR_motors'][motor]['view']:
            aligned, COR_pxl    = cen.COR_from_topview(imagestack, calc_positions, mode, align = True)
        else:
            aligned, COR_pxl = cen.COR_from_sideview(imagestack, calc_positions, mode)
            plot = False
            print('cant plot summaries for this view...')
                                  
                
        dpxl = self.cross_to(COR_pxl[1],COR_pxl[0], view=view, move=False)
        COR_motors = self.stagegeometry['COR_motors'][motor]['motors']
        # updating the absolute position of the COR in motor units:
        for i, COR_mot in enumerate(COR_motors):
            dpos = dpxl[i] / self.calibration[view][COR_mot]
            self.COR[motor][i] = self.wm(COR_mot) + dpos
                       
        # ## useful for debugging
        # print "importing image_tools for saving"
        # from fileIO.images import image_tools as it
        # imagefnametpl = "T:\johannes\COR_images\COR5_img%04d.jpg"
        # for i in range(imagestack.shape[0]):
        #     print "saving image %04d of %s" %(i,imagestack.shape[0])
        #     imagefname = imagefnametpl % i
        #     image = it.optimize_greyscale(imagestack[i])
        #     print image.shape
        #     print image.dtype
        #     it.array_to_imagefile(image,imagefname)        
        # ## until here

        if plot:
            print 'showing results'
            self._show_results(imagestack, aligned, positions, save=saveimages, prefix = saveimages_prefix, COR=COR_pxl)
        
        print 'Done. Found COR at ', self.COR[motor]

        # neccessary cleanup for memmap
        if type(imagestack) == np.core.memmap:
            imagestack_tmpfname = imagestack.filename
            other_tmpfname = aligned.filename
            del imagestack
            del aligned
            gc.collect()
            os.remove(imagestack_tmpfname)
            os.remove(aligned_tmpfname)

        
        return self.COR

    def make_calibration(self,
                         view = 'side',
                         motor = 'y',
                         mode = 'com',
                         step = 0.1,
                         steps = 4,
                         plot = True,
                         backlashcorrection = True,
                         troi = None,
                         cutcontrast=0.5,
                         saveimages = False,
                         axis=1,
                         saveimages_prefix = '',
                         sleep=0):
        '''
        calibrate the movement of a motor against the microscope image 
        in pxl/step
        view = 'side' or 'top'
        using one of the image recogintion modes: 
            'COM'     - center of mass, fastest
            'elastix' - default, includes contribution of second axis
            'CC'      - 1d_cross correlation, slow
            'test'    - does all the modes, really slow
        step = order of magnitute of senible steps to make (will be used for backlashcorrection)
        '''
        ### check that this view makes sense for the motor
        # if view == 'side':
        #     if motor == 'y':
        #         axis = 1
        #     elif motor == 'z':
        #         axis = 0
        #     else:
        #         print 'cannot calibrate motor %s with %s view!' % (motor, view)
        #         return False
        # elif view == 'top':
        #     if motor == 'y':
        #         axis = 0
        #     elif motor == 'x':
        #         axis = 1
        #     else:
        #         print 'cannot calibrate motor %s with %s view!' % (motor, view)
        #         return False
            
        print 'motor %s will be calibrated with a series of images in %s view' % (motor, view)

        if mode.upper() not in ['ELASTIX','COM','CC','TEST']:
            raise NotImplementedError(mode ,' is not a valid image alignment mode for calibration')
        
        pos_ini = np.copy(self.wm(motor))
        positions = [pos_ini + i*step for i in range(steps)]

        if backlashcorrection:
            backlashcorrection = step

        if plot>1:
            plot_imagestack = True
        else:
            plot_imagestack = False
            
            
        imagestack = self._get_imagestack(view = view,
                                          plot = plot_imagestack,
                                          troi = troi,
                                          positions = positions,
                                          motor  = motor,
                                          cutcontrast = cutcontrast,
                                          backlashcorrection = backlashcorrection,
                                          sleep=sleep)
        
        ### start with actual calibration code:

        if mode.upper() == 'TEST':
        ### testing all modes the other elifs are copies of each following boilerplate block of code:
            shift = []

            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'elastix', 'elastix_mode':'translation'}
            dummy, elas_shift = ia.image_align(dummy, mode)
            elas_sum = dummy.sum(0)
            shift = [[positions[i],np.sign(dxdy[axis])*np.sqrt(dxdy[0]**2+dxdy[1]**2)] for i,dxdy in enumerate(elas_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'crosscorrelation_1d', 'axis': axis}
            dummy, CC_shift = ia.image_align(dummy, mode)
            CC_sum = dummy.sum(0)
            shift = [[positions[i],np.sign(dxdy[axis])*np.sqrt(dxdy[0]**2+dxdy[1]**2)] for i,dxdy in enumerate(CC_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration

            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            alignment = np.array([0,0])
            alignment[axis] = 1
            mode  = {'mode':'centerofmass', 'alignment':alignment}
            dummy, COM_shift = ia.image_align(dummy, mode)
            COM_sum = dummy.sum(0)
            shift = [[positions[i],np.sign(dxdy[axis])*np.sqrt(dxdy[0]**2+dxdy[1]**2)] for i,dxdy in enumerate(COM_shift)]
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(CC_shift)]
            # shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(COM_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration

        elif mode.upper() == 'ELASTIX':
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'elastix', 'elastix_mode':'translation'}
            dummy, elas_shift = ia.image_align(dummy, mode)
            elas_sum = dummy.sum(0)
            shift = [[positions[i],np.sign(dxdy[axis])*np.sqrt(dxdy[0]**2+dxdy[1]**2)] for i,dxdy in enumerate(elas_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            
        elif mode.upper() == 'CC':
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'crosscorrelation_1d', 'axis': axis}
            dummy, CC_shift = ia.image_align(dummy, mode)
            CC_sum = dummy.sum(0)
            shift = [[positions[i],np.sign(dxdy[axis])*np.sqrt(dxdy[0]**2+dxdy[1]**2)] for i,dxdy in enumerate(CC_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration

            
        elif mode.upper() == 'COM':            
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            alignment = np.array([0,0])
            alignment[axis] = 1
            mode  = {'mode':'centerofmass', 'alignment':alignment}
            dummy, COM_shift = ia.image_align(dummy, mode)
            COM_sum = dummy.sum(0)
            shift = [[positions[i],np.sign(dxdy[axis])*np.sqrt(dxdy[0]**2+dxdy[1]**2)] for i,dxdy in enumerate(COM_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            
        else:
            raise NotImplementedError('%s is not an implemented mode. Try "test", "CC", "COM" or "elastix"' %mode)
            
        
        print 'found calibration of %s pxl/step' % calibration
        
        self._calibrate(motor, calibration, view = view)

        if plot:
            self._show_results(imagestack, dummy, positions,save=saveimages,prefix = saveimages_prefix,COR=self.cross_pxl[view])

        # neccessary cleanup for memmap
        if type(imagestack) == np.core.memmap:
            imagestack_tmpfname = imagestack.filename
            other_tmpfname = dummy.filename
            del imagestack
            del dummy
            gc.collect()
            os.remove(imagestack_tmpfname)
            os.remove(dummy_tmpfname)
            
        return self.calibration[view][motor]



##### lookuptable functionality
#     TODO: check that this now works in the LUT_Feldberg class
    # def _save_lookup(self, function, savename):
    #     data   = np.zeros(shape = (len(self.lookup[function][function]),len(self.lookup[function].keys())))
    #     unsorted_header = self.lookup[function].keys()
    #     header    = []
    #     header.append(unsorted_header.pop(unsorted_header.index(function)))
    #     header   += unsorted_header
    #     for i, mot in enumerate(header):
    #         data[:,i] = self.lookup[function][mot]
        
    #     save_data.save_data(savename, data, header = header)
        
    # def load_lookup(self, savename):
    #     data, header           =  open_data.open_data(savename)
    #     print "found lookuptable for motor: ", header[0]
    #     print 'using (unsorted) motors ', header[1:]
    #     print data
    #     self.lookup[header[0]] = {}
    #     for i, mot in enumerate(header):
    #         self.lookup[header[0]][mot] = data[:,i]        

    
    def make_lookup(self,
                    motor = 'rotz',
                    view = 'top',
                    positions = [0,1,2,3,4,5],
                    mode = 'com',
                    resolution = None,
                    lookup_motors = None,
                    plot = True,
                    troi = None,
                    cutcontrast=0.5,
                    backlashcorrection = True,
                    savename = None,
                    move_using_lookup=False,
                    saveimages=False,
                    saveimages_prefix='lookup1',
                    sleep=0):
        ''' 
        NOT WORKING with new LUT_Anyber class lookuptabers yet TODO
        creates a lookup table for the COR_motor <motor> which must be a key in self.stagegeometry['COR_motors']
        the lookuptable will contain positions of <motor> between 0 and 360 seperated by <resolution> 
        OR values for the defined list of angles <positions>
        for the motors listed under self.stagegeometry['COR_motors'][<motor>]['motors'] positions minimising the movement of the sample are found using the imagealigment mode 'mode' (see self.calibrate_axis)
        corresponding movement command = self.mv(..., move_using_lookup = True,...)
        '''
        if type(positions) == type(None):
            if type(resolution) == type(None):
                raise ValueError('please define either a <resolution> or a list <positions>')
            else:
                positions = [x*resolution for x in range(int(360/resolution))]

        if mode.upper() not in ['ELASTIX','COM','CC']:
            raise NotImplementedError(mode ,' is not a valid image alignment mode for making a lookup table')

        if lookup_motors == None: # assume the same motors as for COR
            mot0 = self.stagegeometry['COR_motors'][motor]['motors'][0]
            mot1 = self.stagegeometry['COR_motors'][motor]['motors'][1]
        else:
            mot0 = lookup_motors[0]
            mot1 = lookup_motors[1]
            
        print 'will try to get a lookuptable to align rotation in ', motor
        print 'with motors %s (horz) an %s (vert)' %(mot0, mot1)
        print 'viewed from the ', view
        print 'using alignment algorithm ', mode
                
        if plot > 1:
            plot_stack=True
        else:
            plot_stack=False

            
        imagestack = self._get_imagestack(view = view,
                                          plot = plot_stack,
                                          troi = troi,
                                          positions = positions,
                                          motor  = motor,
                                          backlashcorrection = backlashcorrection,
                                          move_using_lookup = move_using_lookup,
                                          sleep=sleep)
    
        if mode.upper() == 'ELASTIX':
            print('WARNING THIS IS NOT TESTED')
            sleep(4)
            mode  = {'mode':'elastix', 'elastix_mode':'translation'}
        elif mode.upper() == 'CC':
            print('WARNING THIS IS NOT TESTED')
            sleep(4)                  
            mode  = {'mode':'crosscorrelation'}
        elif mode.upper() == 'COM':
            alignment = np.array([1,1])
            mode  = {'mode':'centerofmass', 'alignment':alignment}     

        # aleviate memory problems:
        if type(imagestack) == np.core.memmap:
            aligned_fname = 'tmp_aligned.tmp'
            aligned = np.memmap(aligned_fname, dtype=np.float16, mode='w+', shape=imagestack.shape)
            aligned[:] = imagestack[:]
        else:    
            aligned = np.copy(imagestack)
            
        aligned, shift = ia.image_align(aligned, mode)
        shift = np.asarray(shift)

        if motor not in self.lookup.keys():
            self.lookup[motor] = {}
        ## else we assume that all motors are correctly defined in the self.lookup[motor] dict!         

        shift_0 = list(shift[:,0])
        shift_1 = list(shift[:,1])
        
        shift_lookup = {}
        shift_lookup[motor] = positions
        shift_lookup[mot0] = shift_0/self.calibration[view][mot0]
        shift_lookup[mot1] = shift_1/self.calibration[view][mot1]



        if move_using_lookup:
            # we have to add or update the values to the old lookup
            print('updataing old lookuptable')

            self.update_lookup(motor=motor, shift_lookup=shift_lookup, overwrite=False, lookup_motor=[mot0,mot1])
        else:
            self.update_lookup(motor=motor, shift_lookup=shift_lookup, overwrite=True, lookup_motors=[mot0,mot1])
            
        # replaced by update_lookup method!    
        #     old_positions = list(self.lookup[motor][motor])
        #     old_mot0   = list(self.lookup[motor][mot0])
        #     old_mot1   = list(self.lookup[motor][mot1])
                
        #     d0 = np.interp(positions, self.lookup[motor][motor], self.lookup[motor][mot0])
        #     d1 = np.interp(positions, self.lookup[motor][motor], self.lookup[motor][mot1])
        #     new_positions = list(positions)
        #     new_mot0   = list(shift[:,1]/self.calibration[view][mot0] + d0)
        #     new_mot1   = list(shift[:,0]/self.calibration[view][mot1] + d1)


        #     for i, new_theta in enumerate(new_positions):
        #         j = 0
        #         old_theta = old_positions[j]
        #         while new_theta > old_theta:
        #             j+=1
        #             if j==len(old_positions):
        #                 old_positions.append(new_theta)
        #                 old_mot0.append(new_mot0[i])
        #                 old_mot1.append(new_mot1[i])
        #                 break
        #             else:
        #                 old_theta = old_positions[j]
        #         else:
        #             if new_theta == old_theta:
        #                 old_mot0[j]=new_mot0[i]
        #                 old_mot1[j]=new_mot1[i]
        #             else:
        #                 old_positions.insert(j,new_theta)
        #                 old_mot0.insert(j,new_mot0[i])
        #                 old_mot1.insert(j,new_mot1[i])
        #     new_positions = old_positions
        #     new_mot1 = old_mot1
        #     new_mot0 = old_mot0
            
        # else:
        #     # just overwrite the old lookup
        #     print('writing new lookuptable')
        #     new_positions = positions
        #     new_mot1   = shift[:,0]/self.calibration[view][mot1]
        #     new_mot0   = shift[:,1]/self.calibration[view][mot0]
            
        # self.lookup[motor].update({motor: np.asarray(new_positions)})
        # self.lookup[motor].update({mot0: np.asarray(new_mot0)})
        # self.lookup[motor].update({mot1: np.asarray(new_mot1)})

        if type(savename) != type(None):
            self._save_lookup(motor,savename)

        if plot:
            self._show_results(imagestack, aligned, positions, save = saveimages, prefix = saveimages_prefix, COR=self.cross_pxl[view])
            
            plt.matshow(imagestack.sum(0))
            for point in shift[:]:
                plt.plot(-point[1]+self.cross_pxl[view][1],-point[0]+self.cross_pxl[view][0],'bx')
                plt.plot(self.cross_pxl[view][1], self.cross_pxl[view][0],'rx')
            plt.show()


            
        # neccessary cleanup for memmap
        if type(imagestack) == np.core.memmap:
            imagestack_tmpfname = imagestack.filename
            other_tmpfname = aligned.filename
            del imagestack
            del aligned
            gc.collect()
            os.remove(imagestack_tmpfname)
            os.remove(aligned_tmpfname)

        
        return self.lookup[motor]


        
    def shift_lookup_cross_to(self,
                              horz_pxl=None,
                              vert_pxl=None,
                              rotmotor='rot',
                              move=True):
        '''
        NOT WORKING with ne LUT_Anyberg class lookuptables yet TODO
        If the cross is positioned on the COR of the old lookuptable, the new COR will be at the given pixels.
        if <move> the cross is move there.
        '''
        view = self.stagegeometry['COR_motors'][rotmotor]['view']
        COR_motors=self.stagegeometry['COR_motors'][rotmotor]['motors']
        COR_shift = self.cross_to(horz_pxl=horz_pxl,vert_pxl=vert_pxl,view=view,move=move)
        print('Shift in pxl: ',COR_shift)
        for i,mot in enumerate(COR_motors):
            COR_shift[i]*=1.0/self.calibration[view][mot]
        
        self.shift_lookup(rotmotor=rotmotor,
                          COR_shift=COR_shift)
        
        print('Lookuptable not saved. Use self._save_lookup if you want to.')
                        
    ### the foloowing functions are now available under self.lookup.

    # def shift_lookup(self,
    #                  rotmotor='rot',
    #                  COR_shift = [0.1,0.1]):
    #     COR_motors=self.stagegeometry['COR_motors'][rotmotor]['motors']
    #     COR_shift = [float(x) for x in COR_shift]
    #     print('shifting lookuptable for ',rotmotor,' with ',COR_motors,' COR by ',COR_shift)
    #     lookup = self.lookup[rotmotor]

        
    #     if self.stagegeometry['COR_motors'][rotmotor]['invert']:
    #         rot_rad = -lookup[rotmotor]/180.0*np.pi
    #     else:
    #         rot_rad = lookup[rotmotor]/180.0*np.pi
    #     lookup[COR_motors[0]] += COR_shift[0]*np.cos(rot_rad) - COR_shift[1]*np.sin(rot_rad)
    #     lookup[COR_motors[1]] += COR_shift[0]*np.sin(rot_rad) + COR_shift[1]*np.cos(rot_rad)

    
    # def update_lookup(self, motor, shift_lookup, overwrite=False, lookup_motors=None):
    #     '''
    #     if overwrite == False, shift_lookup is added to the old lookup as a relative change as if shift lookup was measured using the coorections of self.lookup as a base correction.
    #     '''
    #     # we have to add or update the values to the old lookup
        
    #     if lookup_motors == None: # assume the same motors as for COR
    #         lookup_motors = self.stagegeometry['COR_motors'][motor]['motors']
 
    #     positions = shift_lookup[motor]

    #     if self.motors[motor]['is_rotation']:
    #         positions=np.asarray(positions)
    #         positions=positions % 360.0

    #     # after mod the list is no longer sorted but it needs to be for the np.interp and my following update of the old lookup
    #     # so now sort:
    #     positions_list=list(positions)
 
    #     together_array = np.zeros(shape = (len(positions), len(lookup_motors)+1))


    #     together_array[:,0]=positions

        
    #     for i,mot in enumerate(lookup_motors):
    #         together_array[:,i+1] = list(shift_lookup[mot])
        
    #     print 'together_array'
    #     print together_array
        
    #     together_list=[]
    #     for i in range(len(positions)):
    #         together_list.append([x for x in together_array[i,:]])
                                 


    #     #together = zip(positions_list, shift_0, shift_1)

    #     sorted_together =  sorted(together_list)

    #     # to make the lookuptable look continuos for np.interp, we add the value for 360.0 and 0.0
    #     if self.motors[motor]['is_rotation']:
    #         if not sorted_together[0][0]==0.0:
    #             if sorted_together[0][0] < 360.0-sorted_together[-1][0]:
    #                 sorted_together.insert(0, [0.0] +  list([x for x in sorted_together[0][1:]]))
    #             else:
    #                 sorted_together.insert(0, [0.0] +  list([x for x in sorted_together[-1][1:]]))

    #         if not sorted_together[-1][0]==360.0:
    #             if sorted_together[0][0] < 360.0-sorted_together[-1][0]:
    #                 sorted_together.append([360.0] + list([x for x in sorted_together[0][1:]]))
    #             else:
    #                 sorted_together.append([360.0] +  list([x for x in sorted_together[-1][1:]]))

    #     print 'sorted_together_list'
    #     print sorted_together
    #     positions = [x[0] for x in sorted_together]
    #     shift = []
    #     for i in  range(len(lookup_motors)):
    #         shift.append(np.asarray([x[i+1] for x in sorted_together]))

    #     print('positions')
    #     print(positions)
    #     print('shift')
    #     print(shift)
            
    #     if not overwrite:
    #         print('updating old lookuptable')
    #         old_positions = list(self.lookup[motor][motor])
    #         old_mots = []
    #         for i,mot in enumerate(lookup_motors):
    #             old_mots.append(list(self.lookup[motor][mot]))
    #             d_i = np.asarray((np.interp(positions, self.lookup[motor][motor], self.lookup[motor][mot])))
    #             s_i = np.asarray(shift[i])
    #             shift[i] = list(d_i + s_i)


    #         for i, new_theta in enumerate(new_positions):
    #             j = 0
    #             old_theta = old_positions[j]
    #             while new_theta > old_theta:
    #                 j+=1
    #                 if j==len(old_positions):
    #                     old_positions.append(new_theta)

    #                     for k in range(len(old_mots)):
    #                         old_mots[k].append(shift[i][k])
    #                     #old_mot1.append(new_mot1[i])
    #                     break
    #                 else:
    #                     old_theta = old_positions[j]
    #             else:
    #                 if new_theta == old_theta:
    #                     for k in range(len(old_mots)):
    #                         old_mots[k] = shift[i][k]

    #                     #old_mot1[j]=new_mot1[i]
    #                 else:
    #                     old_positions.insert(j,new_theta)
    #                     for k in range(len(old_mots)):
    #                         old_mots[k].insert(j,shift[i][k])
    #                     #old_mot1.insert(j,new_mot1[i])

                        
    #         new_positions = old_positions
    #         new_mots=[]
    #         for k in range(len(old_mots)):
    #             new_mots.append(old_mot[k])
    #         #new_mot0 = old_mot0
    #     else:
    #         # just overwrite the old lookup
    #         print('writing new lookuptable')
    #         new_positions = positions
    #         print new_positions
    #         print 'new_positions'
    #         new_mots=[]
    #         for k in range(len(shift)):
    #             new_mots.append(shift[k])
    #             #new_mot1   = shift_1
            
    #     self.lookup[motor].update({motor: np.asarray(new_positions)})
    #     for i,mot in enumerate(lookup_motors):
    #         self.lookup[motor].update({mot: np.asarray(new_mots[i])})
    #     self.lookup[motor].update({motor: np.asarray(new_positions)})

    #     #self.lookup[motor].update({mot1: np.asarray(new_mot1)})

    # def plot_lookup(self, motor='phi'):
    #     lut = self.lookup[motor]
    #     for mot in lut.keys():
    #         if not mot==motor:
    #             dummy, ax1 = plt.subplots(1) 
    #             ax1.set_title('%s vs %s'%(mot,motor))                              
    #             ax1.plot(lut[motor],lut[mot])

    # def initialize_temp_lookup(self,lookup_motor = 'phi', save_motor_list=['x','y','z']):
    #     self.tmp_lookup = {}
    #     self.tmp_lookup.update({lookup_motor:{}})
        
    #     for mot in save_motor_list:
    #         self.tmp_lookup[lookup_motor].update({mot:[]})
    #     self.tmp_lookup[lookup_motor].update({lookup_motor:[]})

    #     print 'ready to save lookup positions for ', save_motor_list

    # def tmp_to_lookup(self,lookupmotor):
    #     '''
    #     overwrites the old lookup with the values from tmp lookup
    #     '''
    #     print 'overwriting lookup for %s with:' % lookupmotor
    #     lookup_motors = []
    #     for mot,values in self.tmp_lookup[lookupmotor].items():
    #         lookup_motors.append(mot)
    #         print mot
    #         print values
                                  
    #     self.update_lookup(motor=lookupmotor, shift_lookup=self.tmp_lookup[lookupmotor], overwrite=True, lookup_motors=lookup_motors)      
          
    # def add_pos_to_templookup(self, lookupmotor):

    #     pos = self.save_position('last_addition_to_tmp_lookup')

    #     tmp_lut = self.tmp_lookup[lookupmotor]
        
    #     for mot in self.tmp_lookup[lookupmotor].keys():
    #         tmp_lut[mot].append(pos[mot])
        
##### some cosmetic plotting



    def _show_results(self, imagestack, aligned, positions,save=False,prefix = '',COR=[10,10]):

        if imagestack.shape[0]<30:
            initialtitle = ['initial ' + str(x) for x in positions]
            pa.plot_array(imagestack, title = initialtitle)
                
            alignedtitle = ['aligned ' + str(x) for x in positions]
            pa.plot_array(aligned, title = alignedtitle)
        
        fig1, ax1 = plt.subplots(1) 
        aligned_sum = aligned.sum(0)
        ax1.matshow(aligned_sum)
        ax1.set_title('Sum of all images, realigned')
        ax1.plot(COR[1],COR[0],'bo')
        
        initial_sum = imagestack.sum(0)
        fig2, ax2 = plt.subplots(1) 
        ax2.matshow(initial_sum)
        ax2.set_title('Sum of all images')
        ax2.plot(COR[1],COR[0],'bo')

        
        if save:
            it.save_series(imagestack,savename_list=[prefix+ x + '.png' for x in initialtitle])
            it.save_series(aligned,savename_list=[prefix+ x + '.png' for x in alignedtitle])
            it.array_to_imagefile(aligned_sum,imagefname=prefix+"aligned_sum.png")
            it.array_to_imagefile(initial_sum,imagefname=prefix+"initial_sum.png")
                                            
        
