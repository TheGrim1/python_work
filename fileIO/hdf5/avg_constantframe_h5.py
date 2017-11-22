from __future__ import print_function
from __future__ import absolute_import
# home: /data/id13/inhouse2/AJ/skript/fileIO/hdf5/do_stuff.py

#global
import os
import sys
import h5py
import numpy as np
import time

# local
from .avg_h5 import avg_h5
from .save_h5 import save_h5
from plot_h5 import plot_h5, plotmany_h5
from .open_h5 import open_h5


def setup_frames(predata, srcname, nfiles):
    print("setting up files") 
    srcpath        = os.path.dirname(srcname)
    srcfname       = os.path.basename(srcname)
    if os.path.exists(srcpath):
        destpath   = os.path.sep.join([srcpath, "restacked%s_%s"%(nfiles,os.path.splitext(srcfname)[0])]) 
        if not os.path.exists(destpath):
            os.mkdir(destpath)
        else:
            destpath   = os.path.sep.join([srcpath, "restacked%s_%s"%(nfiles,time.time())]) 
            os.mkdir(destpath)
    else:
        print("Invalid sourcepath: quitting") 
        sys.exit(1)
        
    destfilelist        = []
    for i in range(predata.shape[0]):
        newfname  = destpath + os.path.sep + "frame%s.h5" % i
        newfile   = h5py.File(newfname,"w")
        newgroup        = newfile.create_group('entry')
        newgroupgroup   = newgroup.create_group('data')
    
        newshape        = (1, predata.shape[1], predata.shape[2])
        newgroupgroup.create_dataset('data', np.zeros(shape=newshape), dtype=np.float32)
                                          
        newfile.close()
      
        destfilelist.append(newfname)

  
    return (destfilelist, destpath)




def main(filelist):
#    print filelist
    averageonly     = True

    i           = 1    
    nfiles      = len(filelist) 
    path        = os.path.dirname(filelist[0])
    
    predata     = open_h5(filelist[0])
    premax      = predata.max()
    treshold    = premax - 100
    print("read first datafile, data has the shape ") 
    print(predata.shape)
    
    print("found max %s, setting threshold for data to %s " % (premax , treshold))

    

#    destfilelist, destpath = setup_frames(predata, srcname = filelist[0], nfiles=nfiles)
    destfilelist, destpath = ([],"")
    nframes     = predata.shape[0]
    data        = np.zeros(shape=predata.shape)
    newdata     = np.zeros(shape=(predata.shape))

    if averageonly: 
        print("average only")
        for fname in filelist:
            print("file %s of %s" % (i,nfiles))
            print(fname)

            i += 1
            data = open_h5(fname, threshold = 65000)

            print("maximum of this file:")
            print(data[:,:,:].max())
            
            
            for frame in  range(nframes):

                print("adding frame %s" % (frame))
                newdata[frame,:,:] += data[frame,:,:]
            
    else:
        ### TODO : 
        hugedata        = np.zeros(shape=(nfiles, nframes,  predata.shape[1], predata.shape[2]))
        print("creating restacked version of the data")
        for fname in filelist:
            print("file %s of %s" % (i,nfiles))
            i += 1
            datafile = open_h5(fname)
        
            hugedata        = np.zeros(shape=(nfiles, nframes,  predata.shape[1], predata.shape[2]))           
            for frame in  range(nframes):
                try:
                    print("adding %s %s" % (fname, frame))
                    datafile["/entry/data/data"].readdirect(data)         
                    newdata[i,:,:] += data[frame,:,:]
                except KeyError:
                    print(" some indexfault in file %s, frame %s" % (destfilelist[frame],frame))
                    nfiles -=1

    newdata = newdata / nfiles
    print("maximum of newdata:")
    print(newdata[:,:,:].max())
#    print "newdata has shape:"
#    print newdata.shape

#    plotmany_h5(newdata)
#    sys.exit(-1)
#            
            
#            data     = avg_h5(data)
#            newfname = os.path.sep.join([os.path.dirname(fname),"averages","avg_"+os.path.basename(fname)]) 

    savepath     = "/data/id13/inhouse5/THEDATA_I5_1/d_2016-09-14_inh_ihhc2970/PROCESS/SESSION22"
    srcfname    = os.path.basename(filelist[0])
    savefname   = srcfname[0:(srcfname.find("data_")+4)]
    fullfname   = os.path.sep.join([savepath, "restacked%s_%s_000000.h5"%(nfiles,savefname)]) 
    print("saving file %s" % fullfname)
    save_h5(newdata, fullfname = fullfname)
#            plot_h5(data, title = newfname)
        

#        except:
#            print "ERROR on %s" % fname
#            pass

    print("    /\ \n   /||\ \n    ||   \n    ||\nThat might have worked") 

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
