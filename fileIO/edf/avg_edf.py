import sys, os
import time
import numpy as np

# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.edf.open_edf import open_edf
from fileIO.plots.plot_array import plot_array as plta
from simplecalc.calc import avg_array

def main(args):
    
    base = np.zeros(shape = open_edf(args[0]).shape)
    n = len(args)

    for fname in args:
        if fname.endswith('.edf'):
            print "reading file %s" %fname
#            print "time before reading = "
#            print time.time()
            frame = open_edf(fname)
#            print "time before reading = "
#            print time.time()
            base = avg_array(base, frame, n)

    plta(base)

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
