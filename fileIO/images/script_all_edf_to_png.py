
from __future__ import print_function
from __future__ import division

import sys, os
import fabio
import subprocess
import numpy as np

sys.path.append('/data/id13/inhouse2/AJ/skript')

import fileIO.images.image_tools as it

def main(args):

    edf_fname_list = [x for x in args if x.find('.edf')]
    
    for edf_fname in edf_fname_list:
        image_fname = os.path.splitext(edf_fname)[0]+'.png'
        it.edf_to_image(edf_fname, image_fname)


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
