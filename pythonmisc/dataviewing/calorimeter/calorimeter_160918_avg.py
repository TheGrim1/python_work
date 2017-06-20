# generic
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os



#local
from savgol import savgol_filter, savgol_coeffs ### NEED NEW SCIPY VERSION (0.14.1 +)
import calorimeter_160918 as cal
## backup:
## cp calorimeter_160918_avg.py /data/id13/inhouse2/AJ/skript/pythonmisc/dataviewing/calorimeter/


def pipe_avg(filelist):
    header,data1      = cal.read_calorimeter_datafiles(filelist)

    header,data1      = cal.avg_data(header,data1)

    title             = "avg over files " +  " ".join(filelist)
#    cal.plotmany(header,data1)

    return (header, data1, title)

def pipe_subtract(header, data1, emptyfile):
    emptyheader,empty = cal.read_own_datafile(emptyfile)           

    header,data1      = cal.subtract_empty(header, data1, empty)
#   header,data1      = cal.subtract_test(header, data1, empty)

    title             = "dT with " + emptyheader[1]  +" and files " +  " ".join(header[1:])

    return (header, data1, title)

def gaussian(length, sigma, mu, normalize = True):

    values      = np.zeros(shape = (length,1))
    axis        = [x - length/2 for x in range(length)]
    if normalize:
        values[:,0] = [math.exp(-((x - mu)**2 / (2.0 * sigma**2)))/ (math.sqrt( 2.0 * math.pi) * sigma) for x in axis]
    else:
        values[:,0] = [math.exp(-((x - mu)**2 / (2.0 * sigma**2))/ (math.sqrt( 2.0 * math.pi) * sigma)) for x in axis]

    return values

def pipe_filter(header, data, filtertype = "gaussian"):
    newheader    = []
    newheader.append(header[0])
    npoints = data.shape[0]


###
# good enough for now

    if filtertype == "gaussian":

        mask               = np.zeros(shape=(npoints,1))
        mask[:,0]          = gaussian(npoints, 3, 0)[:,0]

        newdata            = np.zeros(shape=(data.shape))
        newdata[:,0]       = data[:,0]

        for i in range (1, len(header)):
            newheader.append("gausfil_"+header[i])
            newdata[:,i]       = np.convolve(data[:,i], mask[:,0], mode = "same")


        title = "Gaussian (sigma = 3) filtered data of files " + " ".join(newheader[1:])


    if filtertype == "gaussiantest":
        mask               = np.zeros(shape=(npoints,1))

        newheader.append(header[1])
        newdata        = np.zeros(shape=(data.shape[0],6))
        newdata[:,0] = data[:,0]

        newdata[:,1]       = (data[:,1])

        mask[:,0]          = gaussian(npoints, 2, 0)[:,0]
        newheader.append("mask_"+header[1])
        newdata[:,2]       = mask[:,0]

  
        newheader.append("filtered_"+header[1])
        newdata[:,3]       = np.convolve(data[:,1], mask[:,0], mode = "same")

        mask[:,0]          = gaussian(npoints, 4, 0)[:,0]
        newheader.append("filtered_andunsharp_"+header[1])
        newdata[:,4]       = newdata[:,3] + (newdata[:,3]-np.convolve(data[:,1], mask[:,0], mode = "same"))*0.1

        newheader.append("residue_"+header[1])
        newdata[:,5]       = newdata[:,4] - newdata[:,1]

###
# should be best but requires scipy version 0.14.1 :(

    if filtertype == "sg":
#    newdata = np.zeros(shape=(data.shape))
#    for i in range (1, len(header)):
#        newheader.append("sgfil_"+header[i])
#        newdata[:,i]       = savgol_filter(data[:,i], 5, 2)

        title = "SG Filtered data of files " + " ".join(header)

        raise Exception("not implemented")

    if filtertype == "sgtest":
    ### to see before and after:
        newheader.append(header[1])
        newdata        = np.zeros(shape=(data.shape[0],5))
        newdata[:,0] = data[:,0]

        newdata[:,1]       = (data[:,1])

        newheader.append("sg_9_8_"+header[1])
        newdata[:,2]       = savgol_filter(data[:,1], 11, 10, mode = "constant", cval = 0)

        newheader.append("sg_21_20_"+header[1])
        newdata[:,3]       = savgol_filter(data[:,1], 21, 20, mode = "constant", cval = 0)

        newheader.append("residue_"+header[1])
        newdata[:,4]       = newdata[:,3] - newdata[:,1]


### FFT filter:
# legacy

    if filtertype == "fft":

#    newdata = np.zeros(shape=(data.shape))
#    for i in range (1, len(header)):
#        newheader.append("sgfil_"+header[i])
#        newdata[:,i]       = savgol_filter(data[:,i], 5, 2)



        freqdata           = np.fft.fft(data[:,1])
      
        newheader.append(header[1])
        newdata            = np.zeros(shape=(npoints,5))
        newdata[:,0]       = data[:,0]

        newdata[:,1]       = (data[:,1])

        newheader.append("real_masked_"+header[1])

        mask               = np.zeros(shape=(npoints,1))
        
#        mask[npoints/2 - maskfactor*npoints/2 : npoints/2 + maskfactor*npoints/2,0]=1
        mask[:,0]          = gaussian(npoints, 30, 0, normalize = False)[:,0]
#        mask[(npoints-maskfactor*npoints/2):npoints,0]=1
    
        freqdata.real      = mask[:,0] * freqdata.real

        newdata[:,2]       = np.fft.ifft(freqdata)

        newheader.append("mask_"+header[1])
        newdata[:,3]       = mask[:,0]

        newheader.append("there_and_back"+header[1])
        newdata[:,4]       = (freqdata.real)

        title = "FFT Filtered data of files " + " ".join(header)

    return (newheader, newdata, title)

def pipe_overT(emptyfile, header, data1, timerange):

    emptyheader,empty =  cal.read_own_datafile(emptyfile)           

    header, data1      = cal.change_time_to_temp(empty, header, data1, timerange)

    title             = "dT over T with " + emptyheader[1]  +" and " +  " ".join(header[1:])

    return header, data1, title

def pipe_subtract_background(header, data, background = "const"):
    newheader = []

    if background == "const":
        header, data = cal.subtract_constantbackground(header, data)
        title      = "constant background subtracted of " + " ".join(header[1:])

    else:
        print "no background subtracted!"

        
    return header, data, title

def main(filelist):

### read cal data:
    header, data1 = cal.read_calorimeter_datafiles(filelist)

### read own file
#    header, data1 = cal.read_own_datafile(filelist)

### calculate the average
### this is used mainly for the empty measurements to create the emptyfile needed later:
#
#    header, data1, title = pipe_avg(filelist)
#    cal.save(header ,data1 ,title, path = os.path.dirname(filelist[0]))
###  subtract 
###the emptyfile from the current data1:
#
    emptyfile=["/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/2016_09_16_Calorimeter_AJ1/2016_09_16_AJ1/modified_data/avg_over21empty_000.txt"]
    header, data1, title = pipe_subtract(header, data1, emptyfile)
    cal.save(header ,data1 ,title, path = os.path.dirname(filelist[0]))
### filter:
### the current data (preservs shape)
#
    header, data1, title = pipe_filter(header, data1)
    cal.save(header ,data1 ,title, path = os.path.dirname(filelist[0]))
    header, data1, title = pipe_subtract_background(header, data1, background="const")    
    cal.save(header ,data1 ,title, path = os.path.dirname(filelist[0]))
   
### changing t and T to plot dT over T
### exchanges the time axis for a T axis using emptyfile as a reference curve
# 
    
    emptyfile = ["/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/2016_09_16_Calorimeter_AJ1/2016_09_16_AJ1/modified_data/avg_over21empty_000.txt"]
    tempwindow = "up"
    header2, data2, title2 = pipe_overT(emptyfile, header, data1, tempwindow)
    cal.save(header2 ,data2 ,title2, path = os.path.dirname(filelist[0]))

    tempwindow = "down"
    header, data1, title = pipe_overT(emptyfile, header, data1, tempwindow)
    cal.save(header ,data1, title, path = os.path.dirname(filelist[0]))
### plot:
    cal.plotmany(header, data1)

### save as test:
#   header[1]="test.text"
#   cal.save(header, data1, path = os.path.dirname(filelist[0]))

if __name__ == '__main__':

    args = []
#    print "args received"
#    print sys.argv

    try:
        if len(sys.argv) > 1:
            if sys.argv[1].find("-f")!= -1:
                filename = sys.argv[2]
                print "opening file %s" % filname
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
