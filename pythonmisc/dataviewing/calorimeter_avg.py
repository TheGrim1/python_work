import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plotmany(data1):
# list of lists [[[x1],[y1]],[[x2],etc.]
    
    print (len(data1))
    for i in range(len(data1)):
        print "i = %s" % i
        plt.figure(1)
        plt.subplot(len(data1),1,i)
        plt.plot(data1[i][0], data1[i][1])

    plt.show()


def avg_calorimeter_data(data):
# read all outputs data = [[x0],[yavg]] file smust obviously habe the same timescale

    nptemp   =np.array(temp)
#    print "npdata = " 
#    print nptemp
#    print "npdata averaged = " 

#    print nptemp.sum(0)/nptemp.shape[0]
    avgtemp  = nptemp.sum(0)/nptemp.shape[0]
    

    data.append([time, avgtemp.tolist()])
    


def read_calorimeter_datafiles(src,filelist):
# read all outputs data = [[[x0],[y0]],etc.
    data          = []
    time          = []
    temp          = []

    for fname in filelist:
        if fname.endswith(".txt"):
            try:
                print("reading %s " % os.path.join(src,fname))
                f = open(os.path.join(src,fname),"r")
                cfg=f.readlines()
   
                for i in range(3,len(cfg)):
                    dataline = cfg[i].split()
#                    print dataline
                    time.append(float(dataline[0]))
                    temp.append(float(dataline[1])-float(dataline[8]))
                f.close()
                data.append([time,temp])
                time          = []
                temp          = []
            except IndexError, TypeError: 
                print "Error reading %s" % os.path.join(src,fname)
                
    return data

def save_data(data):
# writes a textfile with contents data (list of lists)
    print "To Do saving"




def main(filelist):
    
    src = ""
    
    data    = read_calorimeter_datafiles(src, filelist)

#    x        = np.arange(0.0,100,1)
#    y        = ((x-100)*x)/100 
#    data1    = [x,y,x,x*2]
    

    plotmany(data)
    




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
        
