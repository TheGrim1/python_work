# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:02:05 2017

@author: OPID13
"""


import sys, os
import matplotlib.pyplot as plt
import numpy as np

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

from fileIO.images.image_tools import optimize_greyscale
from simplecalc.slicing import troi_to_slice
import cameraIO.BaslerGrab as bg
from SpecClient import SpecCommand, SpecMotor
import simplecalc.centering as cen
import simplecalc.image_align as ia
import simplecalc.fitting as fit
import fileIO.plots.plot_array as pa
import fileIO.datafiles.save_data as save_data
import fileIO.datafiles.open_data as open_data

class stage():
    def __init__(self):
        pass
        
    def connect(self,spechost = 'lid13lab1', 
                 specsession = 'motexplore', 
                 timeout = 100):
        print 'connecting to %s' % spechost
        self.spechost    = spechost
        print 'specsession named %s'  % specsession
        self.specsession = specsession
        self.timeout     = timeout
        self.motors      = {}
        self._specversion_update()
        self.COR = None
        
##### handling motors
    def _specversion_update(self):
        self.specversion = self.spechost + ':' + self.specsession
                   
    def _add_motors(self,**kwargs):
        self.motors.update(kwargs)
        for function, name in kwargs.items():
            print 'added motor for %s%s called %s%s in this spec session' %((10-len(function))*' ',function,(10-len(name))*' ',name)
    
    def mvr(self, function, distance, move_in_pxl = False, view = 'side'):
        if move_in_pxl:
            distance = distance / self.calibration[view][function]
        cmd = SpecCommand.SpecCommand('mvr', self.specversion, self.timeout)
        print 'mvr %s %s' %(function, distance)
        cmd(self.motors[function], distance)

    def mv(self, function, distance):
        cmd = SpecCommand.SpecCommand('mv', self.specversion, self.timeout)  
        print 'mv %s %s' %(function, distance)
        cmd(self.motors[function], distance)

    def cross_to(self, horz_pxl=None, vert_pxl=None, view = 'side'):
        if vert_pxl==None:
            vert_pxl=self.cross_pxl[0]
        if horz_pxl==None:
            horz_pxl=self.cross_pxl[1]
            
        dhorz = (self.cross_pxl[1] - horz_pxl)
        dvert = (self.cross_pxl[0] - vert_pxl)
        
        if view == 'side':
            horz_func = 'y'
            vert_func = 'z'
        elif view == 'top':
            horz_func = 'x'
            vert_func = 'y'
        else:
            raise ValueError(view + ' is not a valid view!')
                             
        self.mvr(horz_func, dhorz, move_in_pxl = True, view = view)
        self.mvr(vert_func, dvert, move_in_pxl = True, view = view)
                             
    def wm(self, function):
        # print 'getting position of motor %s'% self.motors[function]
        specmotor = SpecMotor.SpecMotor(self.motors[function], self.specversion)  
        return specmotor.getPosition()

        
    def _backlash(self, function, distance):
        self.mvr(function, -1.0*distance)
        self.mvr(function, distance)

    def _calibrate(self, motor, calibration, view = 'side'):
        self.calibration[view][motor] = calibration
        for motorset in self.stagegeomety['same_calibration'][view]:
            if motor in motorset[0]:
                for i, motor in enumerate(motorset[0]):
                    factor = motorset[1][i]
                    self.calibration[view].update({motor:factor*calibration}) 
        
    def SpecCommand(self, command):
        print 'sending %s the command:'% self.specversion
        print command
        cmd = SpecCommand.SpecCommand(command , self.specversion, self.timeout)  
        cmd()
    
##### handling the cameras
    def initialize_cameras(self, plot=True):
        self.cameras =  bg.initialize_cameras()
        if plot:
            bg.identify_cameras(self.cameras)
            print 'top view = camera 0'
            print 'side view = camera 1'
    
    def switch_cameras(self):
        self.cameras = self.cameas[::-1]
        bg.identify_cameras(self.cameras)
        print 'top view = camera 0'
        print 'side view = camera 1'
    
    def _get_view(self, view='top', troi = None):        
        
        if view.upper() == 'TOP':
            cam_no = 0
        elif view.upper() == 'SIDE':
            cam_no = 1
        try:        
            if troi == None:
                return bg.grab_image(self.cameras[cam_no], bw=True)
            else:
                return bg.grab_image(self.cameras[cam_no], bw=True)[troi_to_slice(troi)]
                
        except AttributeError:
            print 'cameras not properly initialized'
 
        
    def plot(self, view = 'top', title = '', troi = None):
        '''
        view = 'top' or 'side'
        '''
        image0    = self._get_view(view)            
        fig0, ax0 = plt.subplots(1) 
        ax0.imshow(image0)
        ax0.plot(self.cross_pxl[1],self.cross_pxl[0],'rx')
        ax0.set_title(view + ' view ' + title)
        return image0
        
##### centering functionality:
    def _goto_COR(self, motor = 'rotz'):
        for i, COR_mot in enumerate(self.stagegeomety['COR_motors'][motor]):
            self.mv(COR_mot, self.COR[motor][i])
        return 1

    def center_COR(self, 
                   view      = 'side', 
                   plot      = False,
                   troi      = None,
                   thetas    = [x*180.0/10 for x in range(10)],
                   mode      = 'COM',
                   motor     = 'rotz',
                   backlashcorrection = True):
        '''
        moves the COR to the position of the first image
        view   = 'top' or 'side'
        plot   = Boolean(True)
        troi   = ((top,left),(height,width)) or None
        thetas = list of absolute angles to rotate stage to
        mode   = centering algorithm used : 'COM', 'CC', 'elastix'
        backlashcorrection = Bool(True)
        motor  = 'rotz' stage name for the rotation to be centered
        '''
        self.get_COR(view      = view,
                     plot      = plot,
                     troi      = troi,
                     thetas    = thetas,
                     mode      = mode,
                     motor     = motor,
                     backlashcorrection = backlashcorrection)
        
        if self._goto_COR(motor = motor):
            print 'SUCCESS, now move your sample into the focus and repeat until COR is sufficiently aligned.' 
        
    def _get_imagestack(self, motor, view, plot, positions, troi = None, midcontrast = 0.5, backlashcorrection = True):
        print 'prepping images... '
        prep_image = self._get_view(view)
        imagestack = np.zeros(shape = ([len(positions)]+list(prep_image.shape)))

        if backlashcorrection:
            print 'doing backlashcorrection'
            self.mv(motor, positions[0])
            self._backlash(motor,5)
                
        print 'starting rotation...'
        for i, pos in enumerate(positions):

            title = 'frame %s of %s at pos = %s'%(i+1, len(positions), pos)
            print title
            self.mv(motor, pos)
            if plot:
                imagestack[i] = self.plot(view, title, troi = troi)
            else:
                imagestack[i] = self._get_view(view, troi)

        print 'returning %s' %motor
        self.mv(motor, positions[0])

        print 'optimizing image contrast' 
        imagestack = optimize_greyscale(imagestack)
        imagestack = np.where(imagestack < midcontrast*np.max(imagestack),0,imagestack)

        return imagestack
            
    def get_COR(self, 
                view      = 'side', 
                plot      = False,
                troi      = None,
                thetas    = [x*180.0/10 for x in range(10)],
                mode      = 'COM',
                motor     = 'rotz',
                backlashcorrection = True):
        '''
        view   = 'top' or 'side'
        plot   = Boolean(True)
        troi   = ((top,left),(height,width)) or None
        thetas = list of absolute angles to rotate stage to
        mode   = centering algorithm used : 'COM', 'CC', 'elastix'
        backlashcorrection = Bool(True)
        motor  = 'rotz' stage name for the rotation to be centered
        '''
        if mode.upper() not in ['ELASTIX','COM','CC']:
            raise NotImplementedError(mode ,' is not a valid image alignment mode for getting the COR')
                    
        imagestack = self._get_imagestack(view = view,
                                          plot = plot,
                                          troi = troi,
                                          positions = thetas,
                                          motor  = motor,
                                          backlashcorrection = backlashcorrection)
                                          
        print 'calculating COR'
        if view == 'side':
            aligned, COR_pxl    = cen.COR_from_sideview(imagestack, thetas, mode)
        elif view == 'top':
            aligned, COR_pxl    = cen.COR_from_topview(imagestack, thetas, mode, align = True)
        else:
            raise ValueError( '%s is not a valid view' % view)
        
        COR_motors = self.stagegeomety['COR_motors'][motor]
        # updating the absolute position of the COR in motor units:
        for i, COR_mot in enumerate(COR_motors):
            self.COR[motor][i] = self.wm(COR_mot) + \
                                 (COR_pxl[i]/ self.calibration[view][COR_mot])

            
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
        print 'should i plot? : ', (not (mode.upper() == 'COM' and view == 'side'))
        if plot and not (mode.upper() == 'COM' and view == 'side'):
            print 'showing results'
            self._show_results(imagestack, aligned, thetas)
        
        print 'Done. Found COR at ', self.COR
        return self.COR

    def calibrate_axis(self,
                       view = 'side',
                       motor = 'y',
                       mode = 'elastix',
                       plot = True,
                       backlashcorrection = True,
                       step = 0.1,
                       steps = 4,
                       troi = None):
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
        if view == 'side':
            if motor == 'y':
                axis = 1
            elif motor == 'z':
                axis = 0
            else:
                print 'cannot calibrate motor %s with %s view!' % (motor, view)
                return False
        elif view == 'top':
            if motor == 'y':
                axis = 0
            elif motor == 'x':
                axis = 1
            else:
                print 'cannot calibrate motor %s with %s view!' % (motor, view)
                return False
            
        print 'motor %s will be calibrated with a series of images in %s view' % (motor, view)

        if mode.upper() not in ['ELASTIX','COM','CC','TEST']:
            raise NotImplementedError(mode ,' is not a valid image alignment mode for calibration')
        
        steps   = steps
        pos_ini = np.copy(self.wm(motor))
        positions = [pos_ini + i*step for i in range(steps)]

        imagestack = self._get_imagestack(view = view,
                                          plot = plot,
                                          troi = troi,
                                          positions = positions,
                                          motor  = motor,
                                          backlashcorrection = backlashcorrection)
        if plot:
            pa.plot_array(imagestack,title = 'inital')
        ### start with actual calibration code:

        if mode.upper() == 'TEST':
        ### testing all modes the other elifs are copies of each following boilerplate block of code:
            shift = []

            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'elastix', 'elastix_mode':'translation'}
            dummy, elas_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            elas_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(elas_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'crosscorrelation_1d', 'axis': axis}
            dummy, CC_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            CC_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(CC_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration

            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            alignment = np.array([0,0])
            alignment[axis] = 1
            mode  = {'mode':'centerofmass', 'alignment':alignment}
            dummy, COM_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            COM_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(COM_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration

        elif mode.upper() == 'ELASTIX':
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'elastix', 'elastix_mode':'translation'}
            dummy, elas_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            elas_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(elas_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            plt.matshow(elas_sum)
            
        elif mode.upper() == 'CC':
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'crosscorrelation_1d', 'axis': axis}
            dummy, CC_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            CC_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(CC_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            plt.matshow(CC_sum)
            
        elif mode.upper() == 'COM':            
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            alignment = np.array([0,0])
            alignment[axis] = 1
            mode  = {'mode':'centerofmass', 'alignment':alignment}
            dummy, COM_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            COM_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(COM_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            plt.matshow(COM_sum)
            
        else:
            raise NotImplementedError('%s is not an implemented mode. Try "test", "CC", "COM" or "elastix"' %mode)
            
        
        print 'found calibration of %s pxl/step' % calibration
        
        self._calibrate(motor, calibration, view = view)
        
        return self.calibration[view][motor]

##### lookuptable functionality
    def make_lookup(self,
                    motor = 'rotz',
                    resolution = None,
                    thetas = [0,1,2,3,4,5],
                    mode = 'com',
                    plot = True,
                    view = 'top',
                    troi = None,
                    backlashcorrection = True,
                    savename = None):
        ''' 
        creates a lookup table for the COR_motor <motor> which must be a key in self.stagegeomety['COR_motors']
        the lookuptable will contain positions of motor between 0 and 360 seperated by <resolution> 
        OR values for the defined list of angles <thetas>
        for the motors listed under self.stagegeomety['COR_motors'][<motor>] positions minimising the movement of the sample are found using the imagealigment mode 'mode'
        corresponding movement command = self.mv(..., move_using_lookup = True,...)
        '''
        if type(thetas) == type(None):
            if type(resolution) == type(None):
                raise ValueError('please define either a <resolution> or a list <thetas>')
            else:
                thetas = [x*resolution for x in range(int(360/resolution))]

            imagestack = self._get_imagestack(view = view,
                                              plot = plot,
                                              troi = troi,
                                              positions = thetas,
                                              motor  = motor,
                                              backlashcorrection = backlashcorrection)
    
            pa.plot_array(dummy, title = mode['mode'])
            elas_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(elas_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'crosscorrelation_1d', 'axis': axis}
            dummy, CC_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            CC_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(CC_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration

            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            alignment = np.array([0,0])
            alignment[axis] = 1
            mode  = {'mode':'centerofmass', 'alignment':alignment}
            dummy, COM_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            COM_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(COM_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration

        elif mode.upper() == 'ELASTIX':
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'elastix', 'elastix_mode':'translation'}
            dummy, elas_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            elas_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(elas_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            plt.matshow(elas_sum)
            
        elif mode.upper() == 'CC':
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'crosscorrelation_1d', 'axis': axis}
            dummy, CC_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            CC_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(CC_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            plt.matshow(CC_sum)
            
        elif mode.upper() == 'COM':            
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            alignment = np.array([0,0])
            alignment[axis] = 1
            mode  = {'mode':'centerofmass', 'alignment':alignment}
            dummy, COM_shift = ia.image_align(dummy, mode)
            pa.plot_array(dummy, title = mode['mode'])
            COM_sum = dummy.sum(0)
            plt.show()
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(COM_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            plt.matshow(COM_sum)
            
        else:
            raise NotImplementedError('%s is not an implemented mode. Try "test", "CC", "COM" or "elastix"' %mode)
            
        
        print 'found calibration of %s pxl/step' % calibration
        
        self._calibrate(motor, calibration, view = view)
        
        return self.calibration[view][motor]

##### lookuptable functionality
    def _save_lookup(self, savename):
        save_data.save_data(savename, self.lookup, header = ['moved','mot1','mot2'])
        
    def _open_lookup(self, savename):
        lookup, header         =  open_data.open_data(savename)
        self.lookup[header[0]] = lookup
        
    
    def make_lookup(self,
                    motor = 'rotz',
                    resolution = None,
                    thetas = [0,1,2,3,4,5],
                    mode = 'com',
                    plot = True,
                    view = 'top',
                    troi = None,
                    backlashcorrection = True,
                    savename = None):
        ''' 
        creates a lookup table for the COR_motor <motor> which must be a key in self.stagegeomety['COR_motors']
        the lookuptable will contain positions of motor between 0 and 360 seperated by <resolution> 
        OR values for the defined list of angles <thetas>
        for the motors listed under self.stagegeomety['COR_motors'][<motor>] positions minimising the movement of the sample are found using the imagealigment mode 'mode' (see self.calibrate_axis)
        corresponding movement command = self.mv(..., move_using_lookup = True,...) TODO
        '''
        if type(thetas) == type(None):
            if type(resolution) == type(None):
                raise ValueError('please define either a <resolution> or a list <thetas>')
            else:
                thetas = [x*resolution for x in range(int(360/resolution))]

        if mode.upper() not in ['ELASTIX','COM','CC']:
            raise NotImplementedError(mode ,' is not a valid image alignment mode for making a lookup table')
                    

        imagestack = self._get_imagestack(view = view,
                                          plot = plot,
                                          troi = troi,
                                          positions = thetas,
                                          motor  = motor,
                                          backlashcorrection = backlashcorrection)
    
        if mode.upper() == 'ELASTIX':
            mode  = {'mode':'elastix', 'elastix_mode':'translation'}
        elif mode.upper() == 'CC':
            mode  = {'mode':'crosscorrelation'}
        elif mode.upper() == 'COM':
            alignment = np.array([1,1])
            mode  = {'mode':'centerofmass', 'alignment':alignment}     

        aligned = np.copy(imagestack)
        aligned, shift = ia.image_align(aligned, mode)
        if plot:
            pa.plot_array(imagestack, title = 'initial')
            pa.plot_array(imagestack, title = 'aligned')

        self.lookup[motor] = np.zeros(shape = (len(thetas),3))
        self.lookup[:,0]   = thetas
        self.lookup[:,1:3] = shift[:]

        if type(savename) != type(None):
            self._save_lookup(savename,self.lookup[motor])      
                        
        return 1
    
##### some cosmetic plotting
    def _show_results(self, imagestack, aligned, thetas):
        title = ['initial ' + str(x) + 'deg' for x in thetas]
        pa.plot_array(imagestack, title = title)

        title = ['aligned ' + str(x) + 'deg' for x in thetas]
        pa.plot_array(aligned, title = title)

        
        fig1, ax1 = plt.subplots(1) 
        ax1.matshow(aligned.sum(0))
        ax1.set_title('Sum of all images, realigned')        

        fig2, ax2 = plt.subplots(1) 
        ax2.matshow(imagestack.sum(0))
        ax2.set_title('Sum of all images')        
                
        
        
class motexplore_jul17(stage):
    def __init__(self, specsession = 'motexplore'):
        self.connect(specsession = specsession)

        # dictionary connecting function of the motor and its specname:
        motor_dict = {'x'    : 'navix',
                      'y'    : 'naviy',
                      'z'    : 'naviz', 
                      'rotz' : 'srotz'}
        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeomety = {}
        
        # lists of motors that will the rotation axis for centering
        self.stagegeomety['COR_motors'] = {'rotz':['x','y']} # ORDER MATTERS!

        # initializing the default COR at the current motor positions
        self.COR = {}
        [self.COR.update({motor:[self.wm(a),self.wm(b)]}) for motor,[a,b] in self.stagegeomety['COR_motors'].items()]
        
        # lists of motors that have the same calibration:
        # the second list can be a known difference factor, usually useful if it is -1 for eg.
        self.stagegeomety['same_calibration'] = {}
        self.stagegeomety['same_calibration']['side'] = [[['x','y'],[1,1]]]
        self.stagegeomety['same_calibration']['top']  = [[['x','y'],[1,1]]]
        
        
        self.calibration = {}
        self.calibration.update({'side':{}})
        self.calibration.update({'top':{}})
        print 'setting default calibration for zoomed out microscopes'
        self._calibrate('y',-1495.4,'side')
        self._calibrate('y',1495.4,'top')
        self._calibrate('z',-914.02,'side')

        # Point to which the COR is moved:
        self.cross_pxl = [600,1000]

        # lookuptables:
        self.lookup = {}
