# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:02:05 2017

@author: OPID13
"""
from __future__ import print_function
from __future__ import division


import sys, os
import numpy as np
import time

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


import cameraIO.CamView_stages as cvs
import fileIO.images.image_tools as it
import pythonmisc.string_format as sf

PHI_LOOKUP = 'TOMO_phi_top1.dat'
KAPPA_LOOKUP = 'TOMO_kappa_man.dat'

FNAME_TPL = 'drift_docu.dat'

CUTCONTRAST = {'top':-0.95,'side':0.83}

def main():
    stage = cvs.lab_TOMO_navi_sep18()
    stage._calibrate('x',1010,view='top')
    stage._calibrate('x',-1101,view='side')
    stage.lookup.load_lookup(PHI_LOOKUP)
    stage.lookup.load_lookup(KAPPA_LOOKUP)

    stage.mvr('kappa',40,move_using_lookup=True)
    stage.mvr('phi',0,move_using_lookup=True)

    stage.reference_image['side']=np.zeros(shape=stage._get_view('side').shape)
    stage.reference_image['side'][200,320]=100

    counters = ['time [s]']
    [counters.append(x) for x in stage.motors.keys()]
    f = file(fname,'w')    
    f.write(sf.ListToFormattedString(counters,8) +'\r\n') 
    f.close()
    
    for i in range(24*2*60):
        stage.align_to_reference(view='side',
                                 mode='mask_tr',
                                 focus_motor_range=0.1,
                                 focus_points=20,
                                 cutcontrast=CUTCONTRAST['side'])
        f = file(fname,'a')
        data = [time.time()-1536919446]
        [data.append('{:.4f}'.format(stage.wm(x))) for x in counters[1:]]
        f.write(sf.ListToFormattedString(data,8)+'\r\n')
        f.close()
        time.sleep(45)
        


    

        
if __name__=='__main__':
    main()
