from __future__ import print_function
import numpy as np
from silx.io.spech5 import SpecH5
import timeit
import matplotlib.pyplot as plt
import sys,os

sys.path.append('/data/id13/inhouse2/AJ/skript/')
import fileIO.spec.open_scan as my_spec
import simplecalc.fitting as fit

def main(args):
    fname = args[0]
    x_motor = args[1]
    scan_motor = args[2]
    first_scan = args[3]
    no_scans = args[4]
    print(args)
    if len(args)>=6:
        counter = args[5]
    else:
        counter = 'Detector'
    

    
    print('looking at aucau in specsession ', fname)
    print('looking at aucau with %s as x' % x_motor)
    
    print('dscans in ', scan_motor)
    print('%s scans, starting with spec scan number %s' % (no_scans, first_scan))

    
    
    data, x_positions, scan_positions = my_spec.open_dscans(fname,
                                                            scanlist=[x +first_scan for x in range(no_scans)],
                                                            counter=counter,
                                                            sorting_motor=x_motor)
                     
    print('found data.shape', data.shape)
    print('found x positions', x_positions)

   
    fig,ax = plt.subplots()
    fit_results =[]
    for i, scan in enumerate(data):
        ax.plot(scan_positions, scan, label = "%10.3f"%x_positions[i])

        fit_data = np.rollaxis(np.asarray((scan_positions, scan)),-1)
        
        print(fit_data.shape)
        fit_results.append(fit.do_logistic_fit(fit_data, verbose = False))
        print("at %10.3f 2 sigma = %s" , (x_positions[i],2*fit_results[i][3]))
        ax.plot(fit_data[:,0], fit.general_logistic_func(fit_results[i], fit_data[:,0]), "r--", lw = 2)
        
    ax.legend()
    plt.plot()

    results = np.asarray(list(zip(x_positions, [2*beta[3] for beta in fit_results])))
    print('result summary:')
    for bla in results:
        print(bla)
    fig,ax1 = plt.subplots()
    ax1.plot(results[:,0],results[:,1],label = 'caustic %s' %  x_motor)
    plt.show()
    
    return results
    

if __name__ == '__main__':
    
    usage =""" \n1) python <thisfile.py> <aucau x motor> <aucau scan motor> <first aucau scan number> <total number of dscans>
   
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
