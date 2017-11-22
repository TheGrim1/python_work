from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import sys, os
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import simplecalc.gauss_fitting as gauss_fit
from fileIO.datafiles.open_data import open_data
from fileIO.datafiles.save_data import save_data
from simplecalc.linear_combination import do_component_analysis

def main(args):
    '''
    read .dat files with [x, y] data.
    '''
    
    if len(args) > 1:
        print('just give me the folder name')

    ### read all the data
    datadict = {}
    datadir = args[0]

    for fname in os.listdir(datadir):
        if fname.endswith(".dat"):
            fpath = os.path.join(datadir, fname)
            print("reading: %s" % fpath)
            dataset, headerdummy = open_data(fpath, delimiter = ',')
            datadict.update({fname:np.copy(dataset)})
            
    ### homogenize the x and y axis with interpolation
    xmin    = None
    xmax    = None
    xlen    = None
    
    dataheader = list(datadict.keys())
    for fname, dataset in list(datadict.items()):
        if xmin == None:
            xmin = np.min(dataset[:,0])
            xmin = np.max(dataset[:,0])
            xlen = dataset.shape[0]
        else:
            xmin = min(np.min(dataset[:,0]),xmin)
            xmax = max(np.max(dataset[:,0]),xmax)
            xlen = max(dataset.shape[0],xlen)

    dataheader.insert(0,'energy [keV]')
    xaxis = np.atleast_1d(np.arange(xmin,xmax, old_div((float(xmax-xmin)),(2*xlen))))
    fulldata = np.zeros(shape = (len(xaxis), len(dataheader)))
    print(xaxis.shape)
    print(fulldata.shape)
    fulldata[:,0] = xaxis

    for i, fname in enumerate(dataheader[1::]):
        dataset = datadict[fname]
        fulldata[:,i+1] = np.interp(xaxis, dataset[:,0], (dataset[:,1] - dataset[0,1]) )

    print(fulldata)


    ### save the collected data
    savefname = os.path.sep.join([datadir, '../together.dat'])
    save_data(savefname, fulldata, header=dataheader, delimiter='\t')
    
    plt.plot(xaxis,fulldata[:,1::], color = 'black')

    print("looking at the xanes lines")

    xanesfname = '/tmp_14_days/johannes1/lincom/xanes_lines_few.dat'
    xanesdata, xanesheader = open_data(xanesfname, delimiter = '\t')
    xanesheader = xanesheader[0]
    print(xanesheader)
    print(xanesdata.shape)
    xaxis = np.atleast_1d(xanesdata[:,0]*1000)
    plt.plot(xaxis,xanesdata[:,1::])
    plt.show()

    for i in range(len(xanesdata[0,:])-1):
        print('data %s' % xanesheader[i+1])
        beta, residual = do_component_analysis(np.asarray([xaxis,xanesdata[:,i+1]]),fulldata,verbose = True)
        
        
    
    
if __name__ == '__main__':
    
    args = []
    if len(sys.argv) > 1:
        if sys.argv[1].find("-f")== -1:
            f = open(sys.argv[1]) 
            for line in f:
                args.append(line.rstrip())
        else:
            args=sys.argv[2:]
    else:
        f = sys.stdin
        for line in f:
            args.append(line.rstrip())
    
#    print args
    main(args)
