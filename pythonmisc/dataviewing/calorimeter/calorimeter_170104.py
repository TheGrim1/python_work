''' 
Thu Jan  5 15:33:23 CET 2017
updated version of calorimeter_160918.py for faster shell usage
see /data/id13/inhouse2/AJ/skript/pythonmisc/dataviewing/calorimeter/logs/first_for_AJ2a.org
and /data/id13/inhouse2/AJ/skript/pythonmisc/dataviewing/calorimeter/logs/eval_template.org
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
## cp calorimeter_160918.py /data/id13/inhouse2/AJ/skript/pythonmisc/dataviewing/calorimeter/

def plotmany(header, data, path,title = ""):
# list of lists [x1],[y1],[x2],etc.
    ncols = data.shape[1]-1


    plt.figure(1)
    for i in range(ncols):
        plt.subplot(ncols,1,i+1)
        plt.plot(data[:,0], data[:,i+1],label=header[i+1])
#        plt.legend()
#        plt.title(header[i+1])
    plt.xlabel(header[0])
    plt.ylabel("temperature [$^\circ$C] %s" %title)

    savename = os.path.sep.join([path,header[1][0:header[1].find('.txt')]+'.png'])
    print 'saving %s' %savename
    plt.savefig(savename)
    plt.show()

    
def subtract_test(oldheader ,data , empty):
    '''subtract the empty chip response from data and return with same format'''
    
    newdata         = np.zeros(shape=(data.shape[0],4))
    newdata[:,0]    = data[:,0]
    newdata[:, 1]   = data[:,1]
    newheader       = []
    newheader.append(oldheader[0])
    newheader.append(oldheader[1])
    newdata         = np.zeros(shape=(data.shape[0],4))
    newdata[:,0]    = data[:,0]
    newdata[:, 1]   = data[:,1]
    newheader       = []
    newheader.append(oldheader[0])
    newheader.append(oldheader[1])

    newheader.append("emptyavg" + oldheader[1])
    newdata[:,2]    = empty[:,1]

    newheader.append("dT_"+oldheader[1])
    
    try:
        newdata[:,3]    = data[:,1]-empty[:,1]

    except ValueError:
        print "Data and the normalisation (empty chip) don't have the same shape in %s" %(oldheader[1])
        print "Interpolating"
        
        newempty[:,0] = np.interp(data[:,0], empty[:,0], empty[:,1])
        print newempty

        

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
    '''subtract the empty chip response from data and return with same format'''

 
    newdata      = np.zeros(shape=(data.shape))
    newdata[:,0] = data[:,0]
    newheader    = []
    newheader.append(oldheader[0])

    for i in range(1,data.shape[1]):
        newdata[:,i]=data[:,i]-empty[:,1]
        # print "%s - %s = %s"%(data[:,i],empty[:,1],newdata[:,i])
        newheader.append("dT_"+oldheader[i])

    return (newheader,newdata)


def avg_data(oldheader,data):
# get header and data and return avg_over_header and avgdata dim [n,2]    

    avg       = np.zeros(shape=(data.shape[0],2))
    avg[:,0]  = data[:,0]
    ncols     = data.shape[1]-1
    avg[:,1]  = data[:,1:].sum(1)/ncols

    newheader = []
    newheader.append(oldheader[0])
    newheader.append("avg_over%s"%ncols + oldheader[1])

    return newheader,avg

def change_time_to_temp(reference, header, data, upordown):
# timerange [starttime, endtime] in ms

    newheader = ["T_in_C"]

    if upordown == "up":
        timerange = [50, 120]
        for i in range(1,len(header)):
            newheader.append("heating_"+header[i])

    elif upordown == "down":
        timerange = [150, 220]
        for i in range(1, len(header)):
            newheader.append("cooling_"+header[i])

    
    datarange = [x for x in range(data.shape[0]) if data[x,0] >= timerange[0] and data[x,0] <= timerange[1]]
    
    print "datarange = "
    print datarange

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
    

#        save(["time_in_ms", upordown + "temp_in_C"], tandT, title = "translation from timescale on calorimeter run (" + upordown +") to temperature", path="")

#    print "newdata"
#    print newdata

    newdata[:,0] = np.interp(data[datarange,0], tandT[:,0], tandT[:,1])

#    newdata[:,0] = data[datarange,0]
      
    return newheader, newdata

 
def save_data(header,data, title="Title", path= ""):
# save data to file header[1].txt

    if not os.path.exists(path):
        os.mkdir(path)
    
    try:
        savename = os.path.sep.join([path,header[1]])
        print "writing file %s " %savename
        
        f = open(savename, "w")
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
        print "could not write file %s , quitting" %  savename
        sys.exit(0)

    return savename
        
def read_own_datafile(filelist):
    col   = 0
    fname = filelist[0]
    try:
        print("reading processed datafile %s") % (fname)
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
        print "Error reading own datafile %s " % fname

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
            print "Error reading %s " % fname
    
#    print data
    return header,data

def interpolate_empty(empty,data):
    ''' returns empty with len(empty[:,0]) = len(data[:,0]) by interpolating missing values '''
    if not len(data[:,0]) == len(empty[:,0]):
        print "Data and the normalisation (empty chip) don't have the same shape"
        print "Interpolating"
        print empty
        newempty        = np.zeros(shape=((data.shape)[0],2))
        newempty[:,1]   = np.interp(data[:,0], empty[:,0], empty[:,1])
        newempty[:,0]   = data[:,0]
        empty           = newempty
        print empty
    return empty

def do_empty(header,data1,savepath):
    ''' pipe for the empty calibration'''
    
    avgheader, avg = avg_data(header,data1)
    emptyfile      = save_data(avgheader, avg, title="average", path=savepath )
    plotmany(avgheader, avg, path = savepath, title = 'average of empty')
       
    emptyheader,emptyavg= read_own_datafile([emptyfile])

    vartest = subtract_empty(header, data1, emptyavg)
    plotmany(vartest[0],vartest[1], path = savepath, title = 'difference to average of empty')
   
def do_data(header,data1,savepath):
    searchresult =  [empty for empty in os.listdir(savepath) if (empty.find('avg_over')!=-1 and empty.find('_empty')!=-1 and empty.find('.txt')!=-1)]
    if len(searchresult)>1:
        print '\nWARNING\nfound the following possible empty and averaged files:\n'
        print searchresult

    emptyname = searchresult[0]
    emptyfile = os.path.sep.join([savepath,emptyname])
        
    emptyheader,emptyavg = read_own_datafile([emptyfile])
    plotmany(emptyheader,emptyavg,path = savepath,title = 'average of empty')

    emptyavg = interpolate_empty(emptyavg,data1)
    
    dT        = subtract_empty(header, data1, emptyavg)
    save_data(dT[0],dT[1], path = savepath)
    plotmany (dT[0],dT[1], path = savepath, title = 'dT')

    up_over_T = change_time_to_temp(emptyavg, dT[0], dT[1], 'up')
    save_data(up_over_T[0],up_over_T[1],path=savepath)
    plotmany (up_over_T[0],up_over_T[1], path = savepath,title = 'heating')
    
    down_over_T = change_time_to_temp(emptyavg, dT[0], dT[1], 'down')
    save_data(down_over_T[0],down_over_T[1],path=savepath)
    plotmany (down_over_T[0],down_over_T[1], path = savepath, title = 'cooling')
    
def main(filelist):

    savepath     = os.path.sep.join([os.path.dirname(filelist[0]),'eval'])
    header,data1 = read_calorimeter_datafiles(filelist)

    if filelist[0].find('empty')!=-1:
        do_empty(header,data1,savepath)
        
       
    else:
        do_data(header,data1,savepath)
        
    
if __name__ == '__main__':
    
    args = []
#    print "args received"
#    print sys.argv

    try:
        if len(sys.argv) > 1:
            if sys.argv[1].find("-f")!= -1:
                filename = sys.argv[2]
                print "opeining file %s" % filname
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
        print 'usage: python calorimeter.py <files calorimeterdata.txt> \nor include -f to indicate a file contraing the file paths\nor "find anyfile.whatever | python calorimeter.py"'
        sys.exit(1)
#    print "args passed:"
    main(args)
