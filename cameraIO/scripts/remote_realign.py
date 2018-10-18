import sys, os
import numpy as np
import time

importpath = '/data/id13/inhouse2/AJ/skript/'
sys.path.append(importpath)
from cameraIO.CamView_stages import EH2_TOMO_navi_sep18
from fileIO.images.image_tools import array_to_imagefile

CUTCONTRAST = 0.83
CALIBRATION = -1101

LOG_FNAME = "/data/id13/inhouse3/THEDATA_I3_2/d_2018-10-03_inh_ihsc1588_tg/DATA/camera_log/pos_log.dat"
IMAGE_TPL = "/data/id13/inhouse3/THEDATA_I3_2/d_2018-10-03_inh_ihsc1588_tg/DATA/camera_log/img_{:06d}_kappa_{:06d}_phi_{:06d}.png"

def realign(focus=False):

    stage = EH2_TOMO_navi_sep18()
    stage._calibrate('x', CALIBRATION, view='side')

    stage.reference_image['side']=np.zeros(shape=stage._get_view('side').shape)

    max_val = 762
        
    stage.reference_image['side'][230,373]=max_val

    if focus:
        focus_motor_range = 0.1
    
    stage.align_to_reference(view='side',
                             mode='mask_tr',
                             focus_motor_range=focus_motor_range,
                             focus_points=20,
                             cutcontrast=CUTCONTRAST['side'])
    
    timestamp = int(time.time()-1538489070)
    f = file(LOG_FNAME,'a')
    data = [timestamp]
    [data.append('{:.4f}'.format(stage.wm(x))) for x in counters[1:]]
    f.write(sf.ListToFormattedString(data,8)+'\r\n')
    f.close()

    array_to_imagefile(stage._get_view('side'),IMAGE_TPL.format(timestamp,int(kappa*1000), int(phi*1000)))
