from __future__ import print_function
from __future__ import division

import sys, os
import fabio
import subprocess
import numpy as np

sys.path.append('/data/id13/inhouse2/AJ/skript')

import fileIO.images.image_tools as it
from multiprocessing import Pool

def worker(args):
    edf_fname = args[0]
    mask_fname = args[1]
    if mask_fname is None:
        mask = None
    else:
        mask = fabio.open(mask).data
        
    image_fname = os.path.splitext(edf_fname)[0]+'.png'
    it.edf_to_image(edf_fname, image_fname, perc_low=0, perc_high = 99, mask=mask)
    print('saved {}'.format(image_fname))
    
def main(args):

    edf_fname_list = [x for x in args if x.find('.edf')]

    # mask_fname = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-06_inh_ihsc1547_mro/PROCESS/aj_log/mask.edf'
    mask_fname =None

    todo_list = []
    for edf_fname in edf_fname_list:
        todo_list.append([edf_fname, mask_fname])

    worker(todo_list[0])    
    pool=Pool(12)
    pool.map(worker, todo_list)
    pool.close()
    pool.join()
    


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
