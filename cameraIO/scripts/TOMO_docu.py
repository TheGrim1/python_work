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

PHI_LIST = [x*5 for x in range(72)]
KAPPA_LIST = [-x*5 for x in range(10)]

# PHI_LIST=[0,5]
# KAPPA_LIST=[0]

PHI_LOOKUP = 'TOMO_phi_top1.dat'
KAPPA_LOOKUP = 'TOMO_kappa_man.dat'

IMAGEFNAME_TPL = 'C:\\venv\elastix_venv\TOMO_docu2'+os.path.sep + '{}' + os.path.sep + '{}_kappa_{}_phi_{}.png'

CUTCONTRAST = 0.86

def main():
    stage=cvs.lab_TOMO_navi_sep18()

    stage.lookup.load_lookup(PHI_LOOKUP)
    stage.lookup.load_lookup(KAPPA_LOOKUP)

    stage._calibrate('x',1430,view='top')
    stage._calibrate('x',-1493,view='side')
    start_time = time.time()

    stage.make_tmp_lookup(motor='phi',view='side',positions=[x for x in range(360)], mode='topmask_r',focus_motor_range=0.1,focus_points=15,cutcontrast=0.83,correct_vertical=True, move_using_lookup=True)
    stage.lookup.tmp_to_lookup('phi')
    stage.lookup.save_lookup('phi','TOMO_phi_side1.dat')

    stage.reference_image['side']=np.zeros(shape=stage._get_view('side').shape)
    stage.reference_image['side'][200,320]=255
    lut_time = time.time()-make_lut_time

    for kappa_pos in KAPPA_LIST:
        stage.mv('kappa',kappa_pos,move_using_lookup=True)

        for phi_pos in PHI_LIST:
            stage.mv('phi',phi_pos,move_using_lookup=True)

            stage.align_to_reference_image(view='side',
                                           mode='topmask_r',
                                           focus_motor_range=0.2,
                                           focus_points=15,
                                           cutcontrast=CUTCONTRAST)
            
            image_top = stage._get_view('top')
            image_side = stage._get_view('side')

            it.array_to_imagefile(image_top, IMAGEFNAME_TPL.format('lut_corr_top','top',kappa_pos,phi_pos))
            it.array_to_imagefile(image_side, IMAGEFNAME_TPL.format('lut_corr_side','side',kappa_pos,phi_pos))


    lut_corr_time = time.time() - start_time - lut_time
    no_pos= len(KAPPA_LIST)*len(PHI_LIST)

    f = file('TOMO_docu_log.dat','w')
    
    f.write('--'*25+'\r\n')
    f.write('\n\n'+'\r\n')
    f.write('took {} s for to make a full lookuptable with side view lut only'.format(make_lut_time)+'\r\n')
    f.write('this is {} s pp'.format(lut_time/no_pos)+'\r\n')
    f.write('\n\n'+'\r\n')
    f.write('took {} s for {} pos using lut only'.format(lut_time, no_pos)+'\r\n')
    f.write('this is {} s pp'.format(lut_time/no_pos)+'\r\n')
    f.write('\n\n'+'\r\n')
    f.write('took {} s for {} pos using lut and correction only'.format(lut_corr_time, no_pos)+'\r\n')
    f.write('this is {} s pp'.format(lut_corr_time/no_pos)+'\r\n')
    f.write('\n\n'+'\r\n')
    f.write('--'*25+'\r\n')
    
    f.close()
    
if __name__=='__main__':
    main()
    os.system('python Y:\\inhouse\\AJ\\script\\cameraIO\\scripts\\drift_docu.py')
