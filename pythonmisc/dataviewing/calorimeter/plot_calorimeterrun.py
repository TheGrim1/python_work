from __future__ import print_function
from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
## cp calorimeter_160918.py /data/id13/inhouse2/AJ/skript/pythonmisc/dataviewing/calorimeter/

def plotmany(header, data, ylabel = "signal [arb. units]"):
# list of lists [x1],[y1],[x2],etc.
    
    ncols = data.shape[1]-1


    plt.figure(1)
    for i in range(ncols):
        ax1 = plt.subplot(ncols,1,i+1)
        ax1.plot(data[:,0], data[:,i+1],label=header[i+1],linewidth = 2)
#        plt.legend()
#        plt.title(header[i+1])

    ax1.figure.set_size_inches(15,4)

    ax1.set_xlim((50,280))
    ax1.set_xticks([100,150,200,250])
    ax1.set_xticklabels([100,150,200,250],size = 16)
    
    ax1.set_yticks([20, 40, 60, 80, 100, 120])
    ax1.set_yticklabels([20, 40, 60, 80, 100, 120],size = 16)

    ax1.set_xlabel("")
    ax1.set_ylabel("temperature [$^\circ$C]", size = 20)



    plt.savefig("/tmp_14_days/johannes/plot_heatrun.png", transparent = True, bbox_incens = "loose")


    plt.show()

def subtract_test(oldheader ,data , empty):
#subtract the empty chip response from data and return with same format

    try:
        newdata         = np.zeros(shape=(data.shape[0],4))
        newdata[:,0]  = data[:,0]
        newdata[:, 1]  = data[:,1]
        newheader       = []
        newheader.append(oldheader[0])
        newheader.append(oldheader[1])

        newheader.append("emptyavg" + oldheader[1])
        newdata[:,2]    = empty[:,1]

        newheader.append("dT_"+oldheader[1])
        newdata[:,3]    = data[:,1]-empty[:,1]

    except IndexError:
        print("Index Error\n%s (data) and the normalisation (empty chip) don't have the same shape" %(oldheader[1]))

    return (newheader,newdata)

def subtract_constantbackground(oldheader, data):
    newheader = [oldheader[0]]
    newdata=np.zeros(shape=data.shape)
    newdata[:,0]=data[:,0]
    for i in range(1,data.shape[1]):
        newdata[:,i]=data[:,i]-np.average(data[:,i])
#           print "%s - %s = %s"%(data[:,i],empty[:,1],newdata[:,i])
        newheader.append("bkg_"+oldheader[i])
    return newheader, newdata

def subtract_empty(oldheader ,data , empty):
#subtract the empty chip response from data and return with same format

    try:
        newdata      = np.zeros(shape=(data.shape))
        newdata[:,0] = data[:,0]
        newheader    = []
        newheader.append(oldheader[0])

        for i in range(1,data.shape[1]):
            newdata[:,i]=data[:,i]-empty[:,1]
#           print "%s - %s = %s"%(data[:,i],empty[:,1],newdata[:,i])
            newheader.append("dT_"+oldheader[i])

    except IndexError:
        print("Index Error\n%s (data) and the normalisation (empty chip) don't have the same shape" %(oldheader[1]))

    return (newheader,newdata)


def avg_data(oldheader,data):
# get header and data and return avg_over_header and avgdata dim [n,2]    

    avg       = np.zeros(shape=(data.shape[0],2))
    avg[:,0]  = data[:,0]
    ncols     = data.shape[1]-1
    avg[:,1]  = float(data[:,1:].sum(1)/ncols)

    newheader = []
    newheader.append(oldheader[0])
    newheader.append("avg_over%s"%ncols + oldheader[1])

    return newheader,avg

def change_time_to_temp(reference, header, data, upordown):
# timerange [starttime, endtime] in ms

    newheader = ["T_in_C"]

    if upordown == "up":
        timerange = [100, 170]
        for i in range(1,len(header)):
            newheader.append("heating_"+header[i])

    elif upordown == "down":
        timerange = [200, 270]
        for i in range(1, len(header)):
            newheader.append("cooling_"+header[i])

    
    datarange = [x for x in range(data.shape[0]) if data[x,0] >= timerange[0] and data[x,0] <= timerange[1]]
    
    print("datarange = ")
    print(datarange)

    tandT   = np.zeros(shape=(len(datarange),2))
    newdata = np.zeros(shape=(len(datarange),data.shape[1]))
    newdata[:,:] = data[datarange,:]

    tandT[:,0:2] = reference[datarange,0:2]
    if upordown ==  "up":
        tandT[:,1].sort()
    else:
        tandT[:,1].sort()
        tandT[:,1]=tandT[::-1][:,1]

#    print "tandT:"
#    print tandT
    

    outputtimetotempscale = True
    if outputtimetotempscale:
        save(["time_in_ms", upordown + "temp_in_C"], tandT, title = "translation from timescale on calorimeter run (" + upordown +") to temperature", path= "/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/2016_09_16_Calorimeter_AJ1/2016_09_16_AJ1/")

#    print "newdata"
#    print newdata

    newdata[:,0] = np.interp(data[datarange,0], tandT[:,0], tandT[:,1])

#    newdata[:,0] = data[datarange,0]
      
    return newheader, newdata

 
def save(header,data, title="Title", path= ""):
# save data to file header[1].txt

    try:
        print("writing file " + header[1])
        f = open(os.path.sep.join([path,"modified_data",header[1]]), "w")
        try:
            tbw=[]
            tbw.append(title)
            tbw.append("\t".join(header))
            tbw.append("")
            for i in range(data.shape[0]):
                tbw.append("\t".join(([str(np.float64.item(x)) for x in data[i,:] if type(x)==np.float64])))
#                print data[i,:]
#                print "\t".join(([str(np.float64.item(x)) for x in data[i,:] if type(x)==np.float64]))
            f.writelines("%s \n" % l for l in tbw) # Write a sequence of strings to a file

        finally:
            f.close()
    except ValueError:
        print("could not write file %s , quitting" %  os.path.sep.join([path,"modified_data",header[1]]))
        sys.exit(0)

def read_own_datafile(filelist):
    col   = 0
    fname = filelist[0]
    try:
        print(("reading processed datafile %s") % (fname))
        f            = open(fname)
        cfg          = f.readlines()

        col          += 1
            
        filename=os.path.split(fname)[len(os.path.split(fname))-1]


        header       = cfg[1].split()
#       print (header)
        ncols        = len(header)
        length       = len(cfg) 

        data         = np.zeros(shape=(length-3,ncols))

        for i in range (3,length):                    
            dataline = cfg[i].split()
#                    print dataline
            data[i-3,:]=[float(x) for x in dataline]
#                    print data
            
        f.close()
    except IndexError: 
        print("Error reading own datafile %s " % fname)

    return header,data


def read_calorimeter_datafiles(filelist):
    col = 0
    first        = True 
    data=np.zeros(shape=(1,1))

    for fname in filelist:
        try:
            print("reading calorimeter datafile %s" % (fname))
            f            = open(fname)
            cfg          = f.readlines()
            length       = len(cfg)
            col          += 1
            
            filename=os.path.split(fname)[len(os.path.split(fname))-1]

            if first:
                first    = False

                header       =["time[ms]",filename]
                ncols        = len(filelist)
                data         = np.zeros(shape=(length-3,1 + ncols))
 
                for i in range (3,length):                    
                    dataline = cfg[i].split()
#                    print dataline
                    data[i-3,(0,1)]=[(float(dataline[0])),(float(dataline[1]))]
#                    print data
            else:
                header.append(filename) 
                for i in range (3,length):                    
                    dataline  = cfg[i].split()
                    data[i-3,col]=(float(dataline[1]))

            f.close()
        except IndexError: 
            print("Error reading %s " % fname)
    
#    print data
    return header,data

def main(filelist):
#    print "filelist :"
#    print filelist


#    header,data1 = read_calorimeter_datafiles(filelist)
    
    header,data1 = read_own_datafile(filelist)

    plotmany(header,data1)
    


if __name__ == '__main__':
    
    args = []
#    print "args received"
#    print sys.argv

    try:
        if len(sys.argv) > 1:
            if sys.argv[1].find("-f")!= -1:
                filename = sys.argv[2]
                print("opeining file %s" % filname)
                f = open(filename) 
                for line in f:
                    args.append(line.rstrip())
            else:
                args=sys.argv[1:]
        else:
            f = sys.stdin
            for line in f:
                args.append(line.rstrip())
    except:
        print('usage: python calorimeter.py <files calorimeterdata.txt> \nor include -f to indicate a file contraing the file paths\nor "find anyfile.whatever | python calorimeter.py"')
        sys.exit(1)
#    print "args passed:"
    main(args)
