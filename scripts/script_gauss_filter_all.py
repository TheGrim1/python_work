
from __future__ import print_function
from __future__ import division


import sys, os
import numpy as np

from scipy.ndimage.filters import gaussian_filter 
from multiprocessing import Pool

sys.path.append('/data/id13/inhouse2/AJ/skript')

import fileIO.images.image_tools as it
from pythonmisc.worker_suicide import worker_init


def _worker(args):

    fname = args[0]
    sigma1 = args[1]
    sigma2 = args[2]
    savename = args[3]
    
    data = it.imagefile_to_array(fname)
    outdata = gaussian_filter(data, [1,sigma1,sigma2])
    it.array_to_imagefile(outdata, savename, verbose =True)
    

def main(args):

    sigma1 = float(args.pop(0))
    try:
        sigma2 = float(args[0])
        args.pop(0)
    except ValueError:
        simga2 = sigma1               

    args = [os.path.realpath(x) for x in args]
    todo_list = []
    
    for fname in args:
        # print(fname)

        # outdata = interp.zoom(data, [1,zoom,zoom], order=interp_order)
 
        
        # print(outdata.dtype)
        # print(outdata.shape)
        
        fname_list = [os.path.dirname(fname)]+[os.path.sep]+list(os.path.splitext(os.path.basename(fname)))      
        savename = ''.join(fname_list[0:3]+['_filtered',fname_list[-1]])
        
        todo_list.append([fname, sigma1, sigma2, savename])

    
    pool= Pool(12, worker_init(os.getpid()))
    pool.map_async(_worker, todo_list)
    pool.close()
    pool.join()
            
if __name__ == '__main__':
    
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc.  \n2)
python <thisfile.py> -f <file containing args as lines> \n3) find
<*yoursearch* -> arg3 etc.> | python <filter_sigmas> <thisfile.py> """

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

    print(args[:10])
    main(args)
