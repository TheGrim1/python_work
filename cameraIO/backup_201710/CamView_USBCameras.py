from __future__ import print_function

from builtins import object
import cameraIO.BaslerGrab as bg

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
                
        except AttributeError:
            print('cameras not properly initialized') 
