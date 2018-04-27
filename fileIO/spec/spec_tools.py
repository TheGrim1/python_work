
from __future__ import print_function
from __future__ import division
import datetime

import numpy as np

import sys
sys.path.append('/data/id13/inhouse2/AJ/skript')

from silx.io.spech5 import SpecH5 as spech5
from fileIO.images import image_tools as it

def get_scan_starttime(spec_f, scanno):
    if type(spec_f) == str:
        spec_f = spech5(spec_f)
        
    return str(spec_f['{}.1/start_time'.format(scanno)].value)


def get_scan_runtime(spec_f, scanno):
    ''' in seconds'''
    if type(spec_f) == str:
        spec_f = spech5(spec_f)
    epoch_group = '{}.1/measurement/Epoch'.format(scanno)

    try:
        start_epoch = spec_f[epoch_group][0]
        end_epoch = spec_f[epoch_group][-1]
        return end_epoch - start_epoch
    except IndexError:
        return 0
        

    

def print_all_scan_start_times_and_runtime(spec_f, save_fname):
    if type(spec_f) == str:
        spec_f = spech5(spec_f)

    scanno_list = spec_f.keys()
    scanno_list = [int(x.split('.')[0]) for x in scanno_list]

    datalines_list = ['#sacnno scan_starttime scan_runtime[min]' ]
    
    for scanno in scanno_list:
        dataline = []

        dataline.append(str(scanno))
        dataline.append(get_scan_starttime(spec_f,scanno))
        seconds = int(get_scan_runtime(spec_f,scanno))
        minutes = int(seconds/60)
        seconds-= minutes*60
        dataline.append('{}:{:02d}'.format(minutes,seconds))
        dataline.append('\n')
        
        datalines_list.append(' '.join(dataline))

    f = open(save_fname,'w')
    f.writelines(datalines_list)
    

        
def color_images_by_refil():
    path = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-09_commi_blc11352_topup/PROCESS/PTYCHO_SUMMARY/hitmaps/png'
    img_fname_tpl = path +'/a75_{}_pty_test_spir_d__hit1_n_data__0000.png'
    save_fname_tpl = path + '/colored_{}_pty_test_spir_d__hit1_n_data__0000.png'
    img_no_list = range(24,28)
    spec_scanno_list = [x + 44 for x in img_no_list]

    epoch_offset = 53001 - 5*60 - 10 # to refil
    
    spec_f = spech5('/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-09_commi_blc11352_topup/DATA/align/align.dat') 
    
    for img_no, scanno in zip(img_no_list,spec_scanno_list):
        
        
        img = it.imagefile_to_array(img_fname_tpl.format(img_no))


        
        starttime = get_scan_starttime(spec_f, scanno)
        minutes, seconds = [int(x) for x in starttime.split(':')[1:3]]
        
        epoch_img = spec_f['{}.1/measurement/Epoch'.format(scanno)].value.reshape(img[0].shape)
        print(epoch_img[0,0])
        
        mod_img = np.uint8(((epoch_img-epoch_offset)/1200)%3)
        print(img_no, scanno)
        print(starttime)
        print(minutes,seconds)
        print(mod_img)
        
        img[0] = np.where(mod_img==0,0.9*img[0],img[0])
        img[1] = np.where(mod_img==1,0.9*img[1],img[1])
        img[2] = np.where(mod_img==2,0.9*img[2],img[2])

#        img=np.rollaxis(np.rollaxis(img,2,0),2,0)
        print(img.shape)


        it.array_to_imagefile(img, save_fname_tpl.format(img_no))
        

            
