
from __future__ import print_function
from __future__ import division
from multiprocessing import Pool

import sys, os
import numpy as np
import scipy.ndimage.interpolation as interp


sys.path.append('/data/id13/inhouse2/AJ/skript')

from pythonmisc.worker_suicide import worker_init
import fileIO.images.image_tools as it

def _worker(args):

    fname = args[0]
    zoom1 = args[1]
    zoom2 = args[2]
    savename = args[3]
    
    data = it.imagefile_to_array(fname)
    outdata = interp.zoom(data, [1,zoom1,zoom2], order=interp_order)
    it.array_to_imagefile(outdata, savename, verbose =True)
    
def main(args):

    zoom1 = float(args.pop(0))
    try:
        zoom2 = float(args[0])
        args.pop(0)
    except ValueError:
        zoom2 = zoom1
        
    args = [os.path.realpath(x) for x in args]

    todo_list = []        
    for fname in args:
        # print(fname)            

        fname_list = [os.path.dirname(fname)]+list(os.path.splitext(os.path.basename(fname)))
        savename = ''.join(fname_list[0:2]+[os.path.sep]+['_zoom',fname_list[-1]])

        todo_list.append([fname, zoom1, zoom2, savename])

    pool= Pool(12, worker_init)
    pool.map_async(_worker, todo_list)
    pool.close()
    pool.join()
            
    

if __name__ == '__main__':
    
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc.  \n2)
python <thisfile.py> -f <file containing args as lines> \n3) find
<*yoursearch* -> arg1 etc.> | python  <zoom1> <opt. zoom2> <thisfile.py> """
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
