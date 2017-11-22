from __future__ import print_function
from __future__ import absolute_import
from builtins import object
class CamView_grabber(object):
    def __init__(self, **kwargs):
        if 'camera_type' not in kwargs:
            print('WARNIGN: no camera type for this interface defined')
        else:
            camera_type=kwargs['camera_type']

        if camera_type.upper() == 'USB':
            from . import CamView_USBCameras as CVUSB
            self.cameras = CVUSB.USBCameras()
        elif camera_type.upper() == 'ETH':
            if 'cameralist' not in kwargs:
                cameralist=['id13/limaccds/eh2-vlm1','id13/limaccds/eh2-vlm2']
                print('WARNING: Ethernet Baslers need an adress!!')
            else:
                cameralist=kwargs['cameralist']
            from . import CamView_ETHCameras as CVETH
            self.cameras = CVETH.ETHCameras(cameralist=cameralist)
        else:
            print('cameras of type %s not known'%camera_type)

    def grab_image(self, i, troi=None):
        return self.cameras.grab_image(cam_no=i,troi=troi)
    
    def __getitem__(self,i):
        return self.grab_image(cam_no=i,troi=None)
