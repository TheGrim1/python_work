from __future__ import print_function
import sys, os
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import simplecalc.gauss_fitting as gauss_fit
from fileIO.datafiles.open_data import open_data
from fileIO.datafiles.save_data import save_data
from simplecalc.calc import combine_datasets

def main(args):
    '''
    read .dat files with [x, y] data.
    '''

    ### setting up list of files
    filelist = []
    if len(args) > 1:

        print('reading file list')
        for fname in args:
            if fname.endswith(".csv"):
                fpath = os.path.realpath(fname)
                filelist.append(fpath)
                datadir = os.path.dirname(fname)            
    else:
        datadir = args[0]
        for fname in os.listdir(datadir):
            if fname.endswith(".dat"):
                fpath = os.path.realpath(os.path.join(datadir, fname))
                filelist.append(fpath)

    ### do the reading
    datadict = {}
    print(filelist)
    for fpath in filelist:
        print("reading: %s" % fpath)
        fname = os.path.basename(fpath)
        dataset = open_data(fpath, delimiter = ',',quotecharlist = ['#','"'])[0]
        datadict.update({fname:np.copy(dataset)})
        print(('found dataset.shape = ' , dataset.shape))
            
#    print dataset
    ### homogenize the x and y axis with interpolation
    ### moved to external function simplecalc.calc.combine_datasets
    fulldata, dataheader = combine_datasets(datadict)
    dataheader[0] = 'energy [eV]'

    # subtract first value to undo the offset from graphs
#    for i in range(1,len(fulldata[1,:])):
#        fulldata[:,i] += - fulldata[0,i]
    
    savefname = os.path.sep.join([datadir, 'together.dat'])
    save_data(savefname, fulldata, header=dataheader, delimiter='\t')
    print('writing file %s ' %savefname)

    xaxis = np.atleast_1d(fulldata[:,0])
    plt.plot(xaxis,fulldata[:,1::])
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


def do_fluence_ID01():

    
