from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#local imports

import calorimeter_160918 as cal


def ttoT_tick_function(t):

    header, uptandT  = cal.read_own_datafile(["/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/2016_09_16_Calorimeter_AJ1/2016_09_16_AJ1/modified_data/uptemp_in_C.txt"])

    header, downtandT  = cal.read_own_datafile(["/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/2016_09_16_Calorimeter_AJ1/2016_09_16_AJ1/modified_data/downtemp_in_C.txt"])
    
    tandT = np.vstack((uptandT,downtandT))
 
    V = np.interp(t, tandT[:,0], tandT[:,1])

    return [z for z in V]


def Ttot_tick_function(T,upordown):
    
    if upordown == "up":
        header, tandT  = cal.read_own_datafile(["/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/2016_09_16_Calorimeter_AJ1/2016_09_16_AJ1/modified_data/uptemp_in_C.txt"])
    elif upordown == "down":
        header, tandT  = cal.read_own_datafile(["/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/2016_09_16_Calorimeter_AJ1/2016_09_16_AJ1/modified_data/downtemp_in_C.txt"])
        tandT = tandT[::-1]
   
    V = np.interp(T, tandT[:,1], tandT[:,0])

    return [z for z in V]

def plotmany_overT(header, data, upordown ,ylabel = "signal [arb. units]"):
# list of lists [x1],[y1],[x2],etc.

    ncols = data.shape[1]-1
    
    plt.figure(1)

# each line in one subplot

    for i in range(ncols):
        
        ax1 = plt.subplot(ncols,1,i+1)
        
        ax2 = ax1.twiny()

        ax1.plot(data[:,0], data[:,i+1],label=header[i+1])
#        plt.legend()
#        plt.title(header[i+1])
        ax1.set_xlabel("temp [C]")
    
        ax2.set_xlim(ax1.get_xlim())

        ax2.set_xticks(ax1.get_xticks())
        
        ax2.set_xticklabels(Ttot_tick_function(ax1.get_xticks(),upordown))

        ax2.set_xlabel("time [ms]")

    plt.ylabel(ylabel)



    plt.show()


def plotmany_overt(header, data ,ylabel = "signal [arb. units]"):
# list of lists [x1],[y1],[x2],etc.

    ncols = data.shape[1]-1
    
    plt.figure(1)
    
#    plt.title("Calorimetrysignal for multiple heating runs")
    

# all lines in one subplot
    for i in range(ncols):
        
        ax1 = plt.subplot(1,1,1)
        ax1.figure.set_size_inches(15,5)
        ax2 = ax1.twiny()
        ax1.tick_params(axis='y', which='both', labelleft='on', labelright='off')
        
        if i == 7:
            ax1.plot(data[:,0], data[:,i+1]+old_div(i,7.0),label=header[i+1],color = "darkred", linewidth = 2)
        else:
            ax1.plot(data[:,0], data[:,i+1]+old_div(i,7.0),label=header[i+1],color = "darkblue", linewidth = 2)
#        plt.legend()
#        plt.title(header[i+1])
        ax2.set_xlabel("time [ms]",size=20)
        ax2.set_xticklabels([])
    
    print("xlim:")
    ax1.set_xlim((50,280))
    
    ax2.set_xlim(ax1.get_xlim())
 
# at these places a tick will be put in ax2 (in units of ax2):   
    uptempticks = [25.5, 50.5, 70.5, 90.5, 110.5, 125.5]
    downtempticks = [25.5, 50.5, 70.5, 90.5, 110.5, 125.5]
    
# in units of ax1 (needed for the plot)
    uptimeticks = Ttot_tick_function(uptempticks, "up")
    downtimeticks = Ttot_tick_function(downtempticks, "down")
    downtimeticks = downtimeticks[::-1]
    
    x1ticks = uptimeticks + downtimeticks
#    print x1ticks
    ax1.set_xticks(x1ticks)

# converted back to temp as labels > is consistent
    
    x1labels = ttoT_tick_function(x1ticks)
    x1labels = ["%3d" % z for z in x1labels]
#    print x2labels

        
    ax1.set_xticklabels(x1labels, size = 16)


    ax1.set_xlabel("temperature [$^\circ$C]",size=20)
    ax1.set_ylabel("dT [$^\circ$C, shifted]",size=20)
    
 
   

# draw lines at these times:
    framelist  = [60.,81.,82.,83.,84.,121.,158.,159.,160.,161.,182.]

    vertlines = [x * 1.51 for x in framelist]
    ax1.set_ylim(-0.5,3.2)
    ax1.set_yticks([0,0.5,1,1.5,2,2.5,3])
    ax1.set_yticklabels([0,0.5,1,1.5,2,2.5,3],size = 16)
    print(ax1.get_ylim())
    for i in vertlines:
        ax1.plot((i,i), ax1.get_ylim() ,color = "red", linewidth = 0.5)
    
    plt.savefig("/tmp_14_days/johannes/plot_cal.png", transparent = True, bbox_incens = "loose")
   
    plt.show()



def main(filelist):
#    print "filelist :"
#    print filelist


#    header,data1 = read_calorimeter_datafiles(filelist)
    
    header,data1 = cal.read_own_datafile(filelist)
    
    upordown = "up"
    if filelist[0].find("cool") != -1:
        upordown = "down"

    plotmany_overt(header,data1, upordown)
    



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
