from __future__ import print_function

import sys, os

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
import cameraIO.BaslerGrab as bg

from fileIO.images.image_tools import uint8array_to_qimage
from simplecalc.slicing import troi_to_slice

class USBCameras(object):
    def __init__(self):
        self.cameras =  bg.initialize_cameras()
    
    def __getitem__(self,i):
        return self.grab_image(cam_no = i,troi=None)

    def grab_image(self, cam_no, troi=None):
        try:        
            if troi == None:
                return bg.grab_image(self.cameras[cam_no], bw=True)
            else:
                return bg.grab_image(self.cameras[cam_no], bw=True)[troi_to_slice(troi)]
                
        except AttributeError as atre:
            print('cameras not properly initialized')
            print(atre)
        except RuntimeError as rtme:
            print('maybe cameras are allready opened by a different program?')
            print(rtme)

    def grab_qimage(self, cam_no, troi=None):
        array = self.grab_image(cam_no,troi)

        #from https://www.swharden.com/wp/2013-06-03-realtime-image-pixelmap-from-numpy-array-data-in-qt/
        array = numpy.require(array, numpy.uint8, 'C')
        qimage = uint8array_to_qimage(im)
        return qimage
