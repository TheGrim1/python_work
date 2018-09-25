# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:02:05 2017

@author: OPID13
"""
from __future__ import print_function
from __future__ import division


import sys, os
import numpy as np

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


PHI_LOOKUP = 'TOMO_phi_top1.dat'
KAPPA_LOOKUP = 'TOMO_kappa_man2.dat'

FNAME_TPL = 'lut_docu'+os.path.sep + 'lut_{}.dat'

CUTCONTRAST = {'top':-0.95,'side':0.82}

def main():
    stage = cvs.lab_TOMO_navi_sep18()
    stage._calibrate('x',1010,view='top')
    stage._calibrate('x',-1101,view='side')
    stage.lookup.load_lookup(PHI_LOOKUP)
    stage.lookup.load_lookup(KAPPA_LOOKUP)


    for i in range(1):
        stage.lookup.load_lookup(PHI_LOOKUP)
        stage.make_tmp_lookup(motor='phi',
                              view='side',
                              positions=range(360),
                              mode='com',
                              focus_motor_range=0.1,
                              focus_points=17,
                              cutcontrast=CUTCONTRAST['side'],
                              correct_vertical=False,
                              move_using_lookup=True)
        stage.lookup.tmp_to_lookup('phi')
        stage.lookup.save_lookup(FNAME_TPL.format('side_{}'.format(i)))

        stage.lookup.load_lookup(PHI_LOOKUP)
        stage.make_tmp_lookup(motor='phi',
                              view='top',
                              positions=range(360),
                              mode='com',
                              focus_motor_range=None,
                              cutcontrast=CUTCONTRAST['top'],
                              correct_vertical=True,
                              move_using_lookup=True)
        
        stage.lookup.tmp_to_lookup('phi')
        stage.lookup.save_lookup(FNAME_TPL.format('top_{}'.format(i)))
    

        
if __name__=='__main__':
    main()
