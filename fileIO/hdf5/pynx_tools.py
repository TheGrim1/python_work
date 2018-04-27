
from __future__ import print_function
from __future__ import division

import sys, os

import h5py
import numpy as np

sys.path.append('data/id13/inhouse2/AJ/skript')
import fileIO.images.image_tools as it

class pynx_h5_reader(object):
    def __init__(self, h5_fname):
        self.fname = h5_fname

    def get_object(self):
        result_dict = np.load(self.fname)
        return result_dict['obj']
        
    def get_probe(self,probe_no=None):
        result_dict = np.load(self.fname)

        if type(probe_no) == type(None):
            probe_no = 0

        if result_dict['probe'].ndim ==3:
            return result_dict['probe'][probe_no,:,:]
        else:
            return result_dict['probe']

    def get_scan_area_obj(self):
        result_dict = np.load(self.fname)
        return result_dict['scan_area_obj']
        
    def get_scan_area_probe(self):
        result_dict = np.load(self.fname)
        return result_dict['scan_area_probe']
        
    def make_obj_image(self, optimize_greyscale=True):
        data = np.real(self.get_object())
        scan_area =  np.invert(self.get_scan_area_obj())
        if optimize_greyscale:
            data = it.optimize_greyscale(data, mask = scan_area)
        return data
        
        
    def make_probe_image(self, optimize_greyscale=True):
        data = np.real(self.get_probe())[:,:]
        scan_area =  np.invert(self.get_scan_area_probe())
        
        if optimize_greyscale:
            data = it.optimize_greyscale(data, mask = scan_area)
        return data
    
def main(args):

    output_dir = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-09_commi_blc11352_topup/PROCESS/PTYCHO_SUMMARY/ptycho/'
    
    source_fname_list = [x for x in args if x.find('.npz')]
    
 
    obj_fname_list = [output_dir + os.path.dirname(x).split(os.path.sep)[-1] + os.path.splitext(os.path.basename(x))[0]+'_obj.png' for x in source_fname_list]
    probe_fname_list = [output_dir + os.path.dirname(x).split(os.path.sep)[-1] + os.path.splitext(os.path.basename(x))[0]+'_probe.png' for x in source_fname_list]
    for source_fname, obj_fname, probe_fname in zip(source_fname_list,dest_fname_list):
        scan = pynx_h5_reader(source_fname)
        obj = scan.make_obj_image()
        probe = scan.make_probe_image()

        it.array_to_imagefile(obj, obj_fname)
        it.array_to_imagefile(probe, probe_fname)
    


if __name__ == '__main__':
    
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc.  \n2)
python <thisfile.py> -f <file containing args as lines> \n3) find
<*yoursearch* -> arg1 etc.> | python <thisfile.py> """

    args = []
    if len(sys.argv) > 1:
        if sys.argv[1].find("-f")!= -1:
            f = open(sys.argv[2])
            for line in f:
                args.append(line.rstrip())
        else:
            args=sys.argv[1:]
    else:
        f = sys.stdin
        for line in f:
            args.append(line.rstrip())
    
    main(args)

