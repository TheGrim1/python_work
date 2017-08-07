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
from SpecClient import SpecCommand
import simplecalc.centering as cen


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
        self.specversion_update()
        self.say_hello()
        self.COR = None
        
##### handling motors    
    def specversion_update(self):
        self.specversion = self.spechost + ':' + self.specsession
           
    def say_hello(self):
        SpecCommand.SpecCommand('p Hello', self.specversion, self.timeout)
        
    def add_motors(self,**kwargs):
        self.motors.update(kwargs)
        for function, name in kwargs.items():
            print 'added motor for %s%s called %s%s in this spec session' %((10-len(function))*' ',function,(10-len(name))*' ',name)
    
    def mvr(self, function, distance):
        cmd = SpecCommand.SpecCommand('mvr', self.specversion, self.timeout)
        print 'mvr %s %s' %(function, distance)
        cmd(self.motors[function], distance)
    
    def mv(self, function, distance):
        cmd = SpecCommand.SpecCommand('mv', self.specversion, self.timeout)  
        cmd(self.motors[function], distance)
    
    def SpecCommand(self, command):
        print 'sending %s the command:'% self.specversion
        print command
        cmd = SpecCommand.SpecCommand(command , self.specversion, self.timeout)  
        cmd()
    
##### handling the cameras
    def initialize_cameras(self):
        self.cameras =  bg.initialize_cameras()
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
        ax0.set_title(view + ' view ' + title)
        return image0
        
##### centering functionality:
        
    def get_COR(self, 
                view = 'top', 
                plot = True,
                troi = None,
                thetas = [x*180.0/10 for x in range(10)],
                mode = '2D_crosscorrelation',
                backlashcorrection = True):
        '''
        view   = 'top' or 'side'
        plot   = Boolean(True)
        troi   = ((top,left),(height,width)) or None
        thetas = list of absolute angles to rotate stage to
        mode   = centering algorithm used : '2D_centerofmass' 
                                            '2D_crosscorrelation'
        backlashcorrection = Booleas(True)
        '''
        
        print 'prepping images... '
        prep_image = self._get_view(view)
        imagestack = np.zeros(shape = ([len(thetas)]+list(prep_image.shape)))

        if backlashcorrection:
            print 'doing backlashcorrection'
            self.mv('rotz',thetas[0])
            self.mvr('rotz',-5)
            self.mv('rotz',thetas[0])
                
        print 'starting rotation...'
        for i, th in enumerate(thetas):

            title = 'frame %s of %s at %s deg'%(i, len(thetas), th)
            print title
            self.mv('rotz', th)
            if plot:
                imagestack[i] = self.plot(view, title, troi = troi)
            else:
                imagestack[i] = self._get_view(view, troi)
        
        print 'calculating COR' 
        self.COR = cen.COR_from_imagestack(imagestack, thetas, mode)
        
        if plot:
            self._show_results(imagestack, thetas)
        
        print 'Done. Found COR at ', self.COR
        return self.COR
    
##### some cosmetic plotting
    def _show_results(self, imagestack, thetas):
        title = ['at ' + str(x) + 'deg' for x in thetas]
        from fileIO.plots.plot_array import plot_array
        plot_array(imagestack, title = title)
        
        aligned = cen.align_COR(imagestack, thetas, self.COR)
        
        fig0, ax0 = plt.subplots(1) 
        ax0.imshow(aligned[0])
        ax0.plot(self.COR[0],self.COR[1], 'rx')
        ax0.set_title('Image at 0 deg')
        
        
        fig1, ax1 = plt.subplots(1) 
        ax1.imshow(aligned.sum(0))
        ax0.plot(self.COR[0],self.COR[1], 'rx')
        ax1.set_title('Sum of all images, realigned')        

        fig2, ax2 = plt.subplots(1) 
        ax2.imshow(imagestack.sum(0))
        ax0.plot(self.COR[0],self.COR[1], 'rx')
        ax1.set_title('Sum of all images')        
                
        
##### cutomizable stage classes
class navitar_jul17(stage):
    def __init__(self, specsession = 'navitar'):
        self.connect(specsession = specsession)
        motor_dict = {'x'    : 'navix',
                      'y'    : 'naviy',
                      'z'    : 'naviz', 
                      'rotz' : 'srotz'}
        self.add_motors(**motor_dict)
        
        
class motexplore_jul17(stage):
    def __init__(self, specsession = 'motexplore'):
        motor_dict = {'x'    : 'navix',
                      'y'    : 'naviy',
                      'z'    : 'naviz', 
                      'rotz' : 'srotz'}
        self.connect(specsession = specsession)
        self.add_motors(**motor_dict)
        
        
