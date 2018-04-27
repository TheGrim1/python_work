
from __future__ import print_function
from __future__ import division


import sys, os
import numpy as np
import scipy.ndimage.interpolation as interp


sys.path.append('/data/id13/inhouse2/AJ/skript')

import fileIO.images.image_tools as it


def main(args):

    zoom = 10.0
    interp_order = 0
    
    for fname in args:
        data = it.imagefile_to_array(fname)
        outdata = interp.zoom(data, [1,zoom,zoom], order=interp_order)

        print(outdata.dtype)
        print(outdata.shape)
        it.array_to_imagefile(outdata, fname, verbose =True)

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
