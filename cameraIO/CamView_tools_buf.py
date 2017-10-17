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
import fileIO.images.image_tools as it
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
        self._specversion_update()
        self.COR = None
        
##### handling motors
    def _specversion_update(self):
        self.specversion = self.spechost + ':' + self.specsession
                   
    def _add_motors(self,**kwargs):
        self.motors.update(kwargs)
        for function, name in kwargs.items():
            print 'added motor for %s%s called %s%s in this spec session' %((10-len(function))*' ',function,(10-len(name))*' ',name)

    def _correct_with_lookup(self, function, start_pos, end_pos):
        print 'correcting movement of %s with motors:' % function
        if function in self.lookup.keys():
            for mot in self.lookup[function].keys():
                if mot != function:
                    start_correction = np.interp(start_pos, self.lookup[function][function], self.lookup[function][mot])
                    end_correction   = np.interp(end_pos, self.lookup[function][function], self.lookup[function][mot])
                    correction = end_correction-start_correction
                    self.mvr(mot, correction)
        else:
            print 'no lookuptable found for ' , function
                
    
    def mvr(self, function, distance,
            move_in_pxl = False, view = 'side',
            move_using_lookup = False):
        if move_in_pxl:
            distance = distance / self.calibration[view][function]

        start_pos = self.wm(function)

        cmd = SpecCommand.SpecCommand('mvr', self.specversion, self.timeout)
        print 'mvr %s %s' %(function, distance)
        cmd(self.motors[function], distance)

        ### optional correction of motors using lookuptable
        if move_using_lookup:
            end_pos = self.wm(function)
            self._correct_with_lookup(function, start_pos, end_pos)
        
        
        
    def mv(self, function, position,
           move_using_lookup = False):
        start_pos = self.wm(function) # needed for move_using_lookup
        cmd = SpecCommand.SpecCommand('mv', self.specversion, self.timeout)  
        print 'mv %s %s' %(function, position)
        cmd(self.motors[function], position)
        
        ### optional correction of motors using lookuptable
        if move_using_lookup:
            end_pos = self.wm(function)
            self._correct_with_lookup(function, start_pos, end_pos)
            
    def cross_to(self, horz_pxl=None,
                 vert_pxl=None,
                 view = 'side',
                 move=True,
                 move_using_lookup=False):
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

        if move:
            self.mvr(horz_func, dhorz, move_in_pxl = True, view = view, move_using_lookup=move_using_lookup)
            self.mvr(vert_func, dvert, move_in_pxl = True, view = view, move_using_lookup=move_using_lookup)
        else:
            return [dhorz, dvert]
                             
    def wm(self, function):
        # print 'getting position of motor %s'% self.motors[function]
        specmotor = SpecMotor.SpecMotor(self.motors[function], self.specversion)  
        return specmotor.getPosition()

        
    def _backlash(self, function, backlashcorrection):
        if type(backlashcorrection) == bool:
            distance = 5.0
        else:
            distance = backlashcorrection
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
        self.cameras = self.cameras[::-1]
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
        image0    = self._get_view(view,troi=troi)            
        fig0, ax0 = plt.subplots(1) 
        ax0.imshow(image0)
        ax0.plot(self.cross_pxl[1],self.cross_pxl[0],'rx')
        ax0.set_title(view + ' view ' + title)
        return image0
        
##### centering functionality:
    def _goto_COR(self, motor = 'rotz', move_using_lookup=False):
        for i, COR_mot in enumerate(self.stagegeomety['COR_motors'][motor]):
            self.mv(COR_mot, self.COR[motor][i],move_using_lookup=move_using_lookup)
        return 1

    def center_COR(self, 
                   motor     = 'rotz',
                   view      = 'side', 
                   thetas    = [x*180.0/10 for x in range(10)],
                   mode      = 'COM',
                   plot      = False,
                   troi      = None,                   
                   cutcontrast = 0.5,                   
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
        cutcontrast [-1.0:1.0](optional) use a negative value to cut dark values, positive for bright values
        '''
        self.get_COR(view      = view,
                     plot      = plot,
                     troi      = troi,
                     thetas    = thetas,
                     mode      = mode,
                     motor     = motor,
                     cutcontrast = cutcontrast,
                     backlashcorrection = backlashcorrection)
        
        if self._goto_COR(motor = motor):
            print 'SUCCESS, now move your sample into the focus and repeat until COR is sufficiently aligned.' 
        
    def _get_imagestack(self,
                        view,
                        motor,
                        positions,
                        plot,
                        troi = None,
                        cutcontrast = 0.5,
                        move_using_lookup = False,
                        backlashcorrection = True,
                        temp_file_fname = 'tmp_imagestack.tmp'):

        print 'prepping images... '
        prep_image = self._get_view(view,troi=troi)
        print('created temp file: ',temp_file_fname) # this is supposed to aleviate some memory bottlenecks
        #imagestack = np.zeros(shape = ([len(positions)]+list(prep_image.shape)))
        # TODO: do this only if neccessary
        imagestack = np.memmap(temp_file_fname, dtype=np.float16, mode='w+', shape=([len(positions)]+list(prep_image.shape)))
        
        
        if backlashcorrection:
            print 'doing backlashcorrection'
            self.mv(motor, positions[0],move_using_lookup=move_using_lookup)
            self._backlash(motor,backlashcorrection)
                
        print 'starting rotation...'
        for i, pos in enumerate(positions):

            title = 'frame %s of %s at pos = %s'%(i+1, len(positions), pos)
            print title
            self.mv(motor, pos,move_using_lookup=move_using_lookup)
            if plot:
                imagestack[i] = self.plot(view, title, troi = troi)
            else:
                imagestack[i] = self._get_view(view, troi)

        print 'returning %s' %motor
        self.mv(motor, positions[0],move_using_lookup=move_using_lookup)

        print 'optimizing image contrast'

        if cutcontrast > 0:
            print 'cutting low intensities'
            imagestack=np.where(imagestack<0.5*np.max(imagestack),0,imagestack)
            perc_low = 1
            perc_high = 100
        else:
            print 'cutting high intensities'
            imagestack=np.where(imagestack>0.5*np.max(imagestack),np.max(imagestack),imagestack)
            perc_low=0
            prec_high =99

        imagestack = optimize_greyscale(imagestack, perc_low=perc_low, perc_high=perc_high)


        return imagestack
            
    def get_COR(self, 
                motor     = 'rotz',
                view      = 'side', 
                thetas    = [x*180.0/10 for x in range(10)],
                mode      = 'COM',
                plot      = False,
                troi      = None,
                cutcontrast = 0.5,
                move_using_lookup= False,
                backlashcorrection = True,
                saveimages = False,
                saveimages_prefix = ''):
        '''
        view   = 'top' or 'side'
        plot   = Boolean(True)
        troi   = ((top,left),(height,width)) or None
        thetas = list of absolute angles to rotate stage to
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
                                          positions = thetas,
                                          motor  = motor,
                                          cutcontrast = cutcontrast,
                                          move_using_lookup=move_using_lookup,
                                          backlashcorrection = backlashcorrection)
                                          
        print 'calculating COR'
        if view == 'side':
            aligned, COR_pxl    = cen.COR_from_sideview(imagestack, thetas, mode)
        elif view == 'top':
            aligned, COR_pxl    = cen.COR_from_topview(imagestack, thetas, mode, align = True)
        else:
            raise ValueError( '%s is not a valid view' % view)
        
        dpxl = self.cross_to(COR_pxl[1],COR_pxl[0], view=view, move=False)
        COR_motors = self.stagegeomety['COR_motors'][motor]
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
        print 'should i plot? : ', (not (mode.upper() == 'COM' and view == 'side'))
        if plot:
            print 'showing results'
            self._show_results(imagestack, aligned, thetas, save=saveimages, prefix = saveimages_prefix, COR=COR_pxl)
        
        print 'Done. Found COR at ', self.COR

        # neccessary cleanup for memmap
        if type(imagestack) == numpy.core.memmap.memmap:
            todelete = imagestack.filename
            del imagestack
            gc.collect()
            os.remove(todelete)

        
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
                         saveimages_prefix = ''):
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
            elas_sum = dummy.sum(0)
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(elas_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'crosscorrelation_1d', 'axis': axis}
            dummy, CC_shift = ia.image_align(dummy, mode)
            CC_sum = dummy.sum(0)
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
            COM_sum = dummy.sum(0)
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(COM_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration

        elif mode.upper() == 'ELASTIX':
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'elastix', 'elastix_mode':'translation'}
            dummy, elas_shift = ia.image_align(dummy, mode)
            elas_sum = dummy.sum(0)
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(elas_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            
        elif mode.upper() == 'CC':
            dummy = np.copy(imagestack)
            dummy = np.where(dummy < 0.5*np.max(dummy),0,dummy)
            mode  = {'mode':'crosscorrelation_1d', 'axis': axis}
            dummy, CC_shift = ia.image_align(dummy, mode)
            CC_sum = dummy.sum(0)
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(CC_shift)]
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
            shift = [[positions[i],np.sqrt(dx**2+dy**2)] for i,[dx,dy] in enumerate(COM_shift)]
            print mode['mode'] + ' found a shift of ', shift
            calibration = -fit.do_linear_fit(np.asarray(shift),verbose = True)[0]
            print mode['mode'] + ' found calibration of ', calibration
            
        else:
            raise NotImplementedError('%s is not an implemented mode. Try "test", "CC", "COM" or "elastix"' %mode)
            
        
        print 'found calibration of %s pxl/step' % calibration
        
        self._calibrate(motor, calibration, view = view)

        if plot:
            self._show_results(imagestack, dummy, positions,save=saveimages,prefix = saveimages_prefix,COR=self.cross_pxl)

        # neccessary cleanup for memmap
        if type(imagestack) == numpy.core.memmap.memmap:
            todelete = imagestack.filename
            del imagestack
            gc.collect()
            os.remove(todelete)

            
        return self.calibration[view][motor]



##### lookuptable functionality
    def _save_lookup(self, function, savename):
        data   = np.zeros(shape = (len(self.lookup[function][function]),len(self.lookup[function].keys())))
        unsorted_header = self.lookup[function].keys()
        header    = []
        header.append(unsorted_header.pop(unsorted_header.index(function)))
        header   += unsorted_header
        for i, mot in enumerate(header):
            data[:,i] = self.lookup[function][mot]
        
        save_data.save_data(savename, data, header = header)
        
    def load_lookup(self, savename):
        data, header           =  open_data.open_data(savename)
        print "found lookuptable for motor: ", header[0]
        print 'using (unsorted) motors ', header[1:]
        print data
        self.lookup[header[0]] = {}
        for i, mot in enumerate(header):
            self.lookup[header[0]][mot] = data[:,i]        
    
    def make_lookup(self,
                    motor = 'rotz',
                    view = 'top',
                    thetas = [0,1,2,3,4,5],
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
                    saveimages_prefix='lookup1'):
        ''' 
        creates a lookup table for the COR_motor <motor> which must be a key in self.stagegeomety['COR_motors']
        the lookuptable will contain positions of <motor> between 0 and 360 seperated by <resolution> 
        OR values for the defined list of angles <thetas>
        for the motors listed under self.stagegeomety['COR_motors'][<motor>] positions minimising the movement of the sample are found using the imagealigment mode 'mode' (see self.calibrate_axis)
        corresponding movement command = self.mv(..., move_using_lookup = True,...)
        '''
        if type(thetas) == type(None):
            if type(resolution) == type(None):
                raise ValueError('please define either a <resolution> or a list <thetas>')
            else:
                thetas = [x*resolution for x in range(int(360/resolution))]

        if mode.upper() not in ['ELASTIX','COM','CC']:
            raise NotImplementedError(mode ,' is not a valid image alignment mode for making a lookup table')

        if lookup_motors == None: # assume the same motors as for COR
            mot0 = self.stagegeomety['COR_motors'][motor][0]
            mot1 = self.stagegeomety['COR_motors'][motor][1]
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
                                          positions = thetas,
                                          motor  = motor,
                                          backlashcorrection = backlashcorrection,
                                          move_using_lookup = move_using_lookup)
    
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
        if type(imagestack) == numpy.core.memmap.memmap:
            aligned_fname = 'tmp_aligned.tmp'
            aligned = np.memmap(aligned_fname, dtype=np.float16, mode='w+', shape=imagestack.shape)
            aligned[:] = imagestack[:]
        else:    
            aligned = np.copy(imagestack)
            
        aligned, shift = ia.image_align(aligned, mode)
        shift = np.asarray(shift)
        if plot:
            pa.plot_array(imagestack, title = 'initial')
            pa.plot_array(aligned, title = 'aligned')
            plt.matshow(imagestack.sum(0))
            for point in shift[:]:
                plt.plot(-point[1]+self.cross_pxl[1],-point[0]+self.cross_pxl[0],'bx')
                plt.plot(self.cross_pxl[1], self.cross_pxl[0],'rx')
            plt.show()

        if motor not in self.lookup.keys():
            self.lookup[motor] = {}
        ## else we assume that all motors are correctly defined in the self.lookup[motor] dict!         
        
        if move_using_lookup:
            # so we have to maybe add some new theta values to the old list
            # this becomes neccessary to reduce the memory load for handling 100+images in one go
            print('updataing old lookuptable')
            old_thetas = list(self.lookup[motor][motor])
            old_mot0   = list(self.lookup[motor][mot0])
            old_mot1   = list(self.lookup[motor][mot1])
                
            d0 = np.interp(thetas, self.lookup[motor][motor], self.lookup[motor][mot0])
            d1 = np.interp(thetas, self.lookup[motor][motor], self.lookup[motor][mot1])
            new_thetas = list(thetas)
            new_mot0   = list(shift[:,1]/self.calibration[view][mot0] + d0)
            new_mot1   = list(shift[:,0]/self.calibration[view][mot1] + d1)


            for i, new_theta in enumerate(new_thetas):
                j = 0
                old_theta = old_thetas[j]
                while new_theta > old_theta:
                    j+=1
                    old_theta = old_thetas[j]
                else:
                    if new_theta == old_theta:
                        old_mot0[j]=new_mot0[i]
                        old_mot1[j]=new_mot1[i]
                    else:
                        old_thetas.insert(j,new_theta)
                        old_mot0.insert(j,new_mot0[i])
                        old_mot1.insert(j,new_mot1[i])
            new_thetas = old_thetas
            new_mot1 = old_mot1
            new_mot0 = old_mot0
            
        else:
            # just overwrite the old lookup
            print('writing new lookuptable')
            new_thetas = thetas
            new_mot1   = shift[:,0]/self.calibration[view][new_mot1]
            new_mot0   = shift[:,1]/self.calibration[view][new_mot0]
            
        self.lookup[motor].update({motor: np.asarray(new_thetas)})
        self.lookup[motor].update({mot0: np.asarray(new_mot0)})
        self.lookup[motor].update({mot1: np.asarray(new_mot1)})

        if type(savename) != type(None):
            self._save_lookup(motor,savename)
        if saveimages:
            self._show_results(imagestack, aligned, thetas, save = True, prefix = saveimages_prefix, COR=self.cross_pxl)

        # # neccessary cleanup for memmap
        # if type(imagestack) == numpy.core.memmap.memmap:
        #     todelete = imagestack.filename
        #     del imagestack
        #     gc.collect()
        #     os.remove(todelete)
        #     todelete = aligned.filename
        #     del aligned
        #     gc.collect()
        #     os.remove(todelete)
            
        return self.lookup[motor]
    
##### some cosmetic plotting
    def _show_results(self, imagestack, aligned, thetas,save=False,prefix = '',COR=[10,10]):
        initialtitle = ['initial' + str(x) + 'deg' for x in thetas]
        pa.plot_array(imagestack, title = initialtitle)
                
        alignedtitle = ['aligned' + str(x) + 'deg' for x in thetas]
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
                                            
        
        
class motexplore_jul17(stage):
    def __init__(self, specsession = 'motexplore', initialize_cameras = True):
        if initialize_cameras:
            self.initialize_cameras(plot=False)
        
        # dictionary connecting function of the motor and its specname:
        self.motors      = {}
        motor_dict = {'x'    : 'navix',
                      'y'    : 'naviy',
                      'z'    : 'naviz', 
                      'rotz' : 'srotz'}
        self._add_motors(**motor_dict)

        # contains the definition of the stage geometry:
        self.stagegeomety = {}
        
        # lists of motors that will the rotation axis for centering
        # self.stagegeomety['COR_motors'] = {'rotation':['y','x']} # ORDER MATTERS!
        self.stagegeomety['COR_motors'] = {'rotz':['x','y']} # ORDER MATTERS!

        # connect to spec
        self.connect(specsession = specsession)
        # initializing the default COR at the current motor positions
        self.COR = {}
        [self.COR.update({motor:[self.wm(a),self.wm(b)]}) for motor,[a,b] in self.stagegeomety['COR_motors'].items()]
        
        # lists of motors that have the same calibration:
        # the second list can be a known difference factor, usually useful if it is -1 for eg.
        self.stagegeomety['same_calibration'] = {}
        self.stagegeomety['same_calibration']['side'] = [[['x','y'],[1,1]]]
        self.stagegeomety['same_calibration']['top']  = [[['x','y'],[-1,-1]]]
     
        self.calibration = {}
        self.calibration.update({'side':{}})
        self.calibration.update({'top':{}})
        print 'setting default calibration for zoomed out microscopes'
        self._calibrate('y',-1495.4,'side')
        self._calibrate('y',1495.4,'top')
        self._calibrate('z',-914.02,'side')

        # General point of reference
        self.cross_pxl = [600,1000]

        # lookuptables:
        self.lookup = {}
        
        # lookuptables look like this:
        # self.lookup[motor] = {} # look up dict for <motor>
        # self.lookup[motor].update({motor: thetas})  ## IMPORTANT, all defined lookup positions are referenced to the SAME positions for <motor>, here <thetas>
        # self.lookup[motor].update({mot0: shift[:,0]*self.calibration[view][mot0]})
        # self.lookup[motor].update({mot1: shift[:,1]*self.calibration[view][mot1]})

        ## else we assume that all motors are correctly defined in the self.lookup[motor] dict!

