from __future__ import print_function
from __future__ import absolute_import
# home: /data/id13/inhouse2/AJ/skript/fileIO/hdf5/do_stuff.py

#global
import os
import sys

# local
from .avg_h5 import avg_h5
from .save_h5 import save_h5
from plot_h5 import plot_h5, plotmany_h5
from .open_h5 import open_h5

def main(filelist):
#    print filelist
    i      = 1    
    nfiles = len(filelist) 
    for fname in filelist:
        try:
            print("averaging %s" %fname)
            print("%s of %s" % (i,nfiles))
            i += 1
            data     = open_h5(fname)
#        plot_h5(data, index = 1)

            data     = avg_h5(data)
            newfname = os.path.sep.join([os.path.dirname(fname),"averages","avg_"+os.path.basename(fname)]) 
            save_h5(data, fullfname = newfname)
#            plot_h5(data, title = newfname)
        except:
            print("ERROR on %s" % fname)
            pass

if __name__ == '__main__':

    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc. 
\n2) python <thisfile.py> -f <file containing args as lines> 
\n3) find <*yoursearch* -> arg1 etc.> | python <thisfile.py> 
"""

  
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
    
#    print args
    main(args)
