# home: /data/id13/inhouse2/AJ/skript/fileIO/hdf5/save_h5.py

import sys, os
import h5py
import numpy as np
from nexusformat.nexus import *

# local import for testing:
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.hdf5.open_h5 import open_h5
import time


def nexus_basicwriter_test(dataset = None, fullfname='/tmp_14_days/johannes1/test2_nexus_hdf5.h5', title = 'default'):

    print "Write a NeXus HDF5 file"

    xaxis = np.atleast_1d(np.arange(100)/100.0*np.pi)
    yaxis = np.atleast_1d(np.arange(200)/200.0*np.pi)
    zdata = np.zeros(shape = (len(yaxis),len(xaxis)))
    zdata[:,:] = np.sin(xaxis)
    zdata *=np.cos(yaxis).reshape(len(yaxis),1)

    zdata2 = np.zeros(shape = (len(yaxis),len(xaxis)))
    zdata2[:,:] = np.sin(2*xaxis)
    zdata2 *=np.cos(5*yaxis).reshape(len(yaxis),1)
 
    # data1
    # create the HDF5 NeXus file
    f = h5py.File(fullfname, "w")
    # point to the default data to be plotted
    f.attrs['default']          = 'entry'
    # give the HDF5 root some more attributes
    f.attrs['file_name']        = fullfname
    f.attrs['creator']          = 'save_h5.py'
    f.attrs['NeXus_version']    = '4.3.0 ... where to find?'
    f.attrs['HDF5_Version']     = h5py.version.hdf5_version
    f.attrs['h5py_version']     = h5py.version.version

    # create the NXentry group
    nxentry = f.create_group('entry')
    nxentry.attrs['NX_class'] = 'NXentry'
    nxentry.attrs['default'] = 'title'
    nxentry.create_dataset('title', data = title)

    # create the NXentry group ############################################
    nxdata = nxentry.create_group('common')
    nxdata.attrs['NX_class'] = 'NXdata'
    nxdata.attrs['signal'] = 'x'       # Y axis of default plot
    nxdata.attrs['axes'] = 'x','y'     # X axis of default plot

    # X axis data
    ds = nxdata.create_dataset('x', data=xaxis)
    ds.attrs['units'] = 'm'
    ds.attrs['long_name'] = '%s in %s' % (ds.name.split("/")[-1],ds.attrs['units']) # suggested plot label
    
    # Y axis data
    ds = nxdata.create_dataset('y', data=yaxis)
    ds.attrs['units'] = 'm'
    ds.attrs['long_name'] = '%s in %s' % (ds.name.split("/")[-1],ds.attrs['units']) # suggested plot label

    
    # create the NXentry group ###########################################
    nxdata = nxentry.create_group('z1')
    nxdata.attrs['NX_class'] = 'NXdata'
    nxdata.attrs['signal'] = 'z'       # Y axis of default plot
    nxdata.attrs['axes'] = 'x','y'     # X axis of default plot

    # Z data
    ds = nxdata.create_dataset('z', data=zdata)
    ds.attrs['units'] = 'cps'
    ds.attrs['long_name'] = '%s in %s' % (ds.name.split("/")[-1],ds.attrs['units']) # suggested plot label
    
    f[ds.parent.name+'/x'] = h5py.SoftLink('/entry/common/x')
    f[ds.parent.name+'/y'] = h5py.SoftLink('/entry/common/y')


    # create the NXentry group ##########################################
    nxdata = nxentry.create_group('z2x5')
    nxdata.attrs['NX_class'] = 'NXdata'
    nxdata.attrs['signal'] = 'z25'       # Y axis of default plot
    nxdata.attrs['axes'] = 'x','y'     # X axis of default plot

    # Z2 data
    ds = nxdata.create_dataset('z25', data=zdata2)
    ds.attrs['units'] = 'cps'
    ds.attrs['long_name'] = '%s in %s' % (ds.name.split("/")[-1],ds.attrs['units']) # suggested plot label

    f[ds.parent.name+'/x'] = h5py.SoftLink('/entry/common/x')
    f[ds.parent.name+'/y'] = h5py.SoftLink('/entry/common/y')
    
    
    f.close()   # be CERTAIN to close the file

def save_h5_nexus(dataset, fullfname, group = 'entry/data', dataname = 'data'):
    ''' saves dataset as .h5 in the nexusformat
    \n TODO fails if groupname/dataname allready exists
    \n TODOif dataset.ndim == 3 and type(dataname) == list and len(dataset[0,0,:]) == len(dataname): 
    \n creates a new dataset per item in dataname and array in dataset.
    '''
    
    pass

    
def save_h5(dataset, fullfname, group = 'entry/data', dataname= 'data'):
    ''' saves dataset in .h5 file fullname.
    \n Creates or opens groups but fails if groupname/dataname allready exists 
    \n if dataset.ndim == 3 and type(dataname) == list and len(dataset[0,0,:]) == len(dataname): 
    \n creates a new dataset per item in dataname and array in dataset.
    '''
    fullfname = os.path.realpath(fullfname)
    savedir   = os.path.dirname(fullfname)
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        print "making directory %s" % savedir
    else:
        if not os.path.exists(fullfname):
            savefile         = h5py.File(fullfname,"w")
            savegroup        = savefile.create_group(group)
            print 'creating group %s' % group
#    print h5dataset.shape
        else: # file allready exists
            savefile         = h5py.File(fullfname,"a")
            try:
                savegroup = savefile.create_group(group)            
            except ValueError:
                savegroup = savefile[group]
                print 'group allready exists'
                
    if type(dataname) == list:
#       print 'listsaving'
        
        if len(dataset[0,0,:]) == len(dataname):
#            print 'dimensions match up'
            
            for i, name in enumerate(dataname):
                savegroup.create_dataset(name, data = dataset[:,:,i], compression = "lzf", shuffle = True)
#                print 'saving dataset as %s' % name       
    else:
        #bulk saving
        savegroup.create_dataset(dataname, data = dataset, compression = "lzf", shuffle = True)
        print 'saving dataset as %s' % dataname  
     

    savefile.flush()
    savefile.close()

    return True

def main(filelist):
## test saving and opening

    fname     = filelist[0]
    starttime = time.time()
    print 
    data      = open_h5(fname)
    opentime  = time.time()-starttime
    print('time to open %s'%opentime)


    savedir   = os.path.dirname(fname)
    savefname = "test_" + os.path.basename(fname)
    if save_h5(data,os.path.sep.join([savedir,savefname])):
        print "hoorray"
               
    print('time to save %s'%(time.time()-starttime - opentime))

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
