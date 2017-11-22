import sys
import os
import commands
import numpy as np

hostname = commands.getoutput('hostname')
print "hostname:", hostname
if hostname not in  ["coherence","cristal","nanofocus"]:
    print('illegal host name')
    sys.exit(1)

import baslertools2


sys.path.append('/data/id13/inhouse2/AJ/skript')
from simplecalc.slicing import troi_to_slice


class ETHCameras(object):
    def __init__(self, cameralist):
        self.cameras = cameralist

    def __getitem__(self,i):
        return self.grab_image(cam_no=i,troi=None)

    def grab_image(self, cam_no, troi=None):
        devname = self.cameras[cam_no]
        cp = baslertools.CameraProxy(devname=devname)
        cp.set_live()
        cp.show_devinfo()
        arr, num = cp.acquire_greyscale_int18() # returns int16 numpy array
        if type(troi)==type(None):
            return arr
        else:
            return arr[troi_to_slice(troi)]

    def grab_qimage(self, cam_no):
        devname = self.cameras[cam_no]
        cp = baslertools.CameraProxy(devname=devname)
        errflg, qimage, last_img_num = self.cp.acquire_qimage()
    
if __name__ == '__main__':
    example()
