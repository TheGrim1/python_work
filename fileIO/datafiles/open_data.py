from __future__ import print_function
import numpy as np
import sys, os
import fakenews
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = fakenews.Dummy()

def open_data(filename, delimiter = ' ', quotecharlist= ['#'],verbose = False):
    '''reads <filename> as a <delimiter> seperated datafile and returns the data as np.array \n ignores lines staring with something in quotecharlist '''
    
    data = []
    f = open(filename, 'r')
    reader = f.readlines()
    
    if verbose:
        print('read lines:')

    header = []
    for i, l in enumerate(reader):
        try:            
            if l.lstrip()[0] in quotecharlist:
                header.append((l[1:].rstrip().split(delimiter)))
            else:
                # print 'found line '
                # print l
                # print l.lstrip().split(delimiter)
                # print 'parsed it as :'
                # print [float(x) for x in (l.rstrip().split(delimiter)) if len(x)> 0]
                data.append([float(x) for x in (l.rstrip().split(delimiter)) if len(x)> 0])
        except IndexError:
            if verbose:
                print('discarding:')
                print(l)

    if verbose:
        print(data)

    if len(header) ==1:
        header = header[0]
            
    data = np.asarray(data)
    return data, header

def main(filenames):

    for filename in filenames:
        data = open_data(filename)

        plt.plot(data)
        plt.show()
    
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
