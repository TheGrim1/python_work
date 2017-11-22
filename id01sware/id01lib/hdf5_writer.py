from __future__ import print_function
from __future__ import division
# script for reading edf into hdf5 using xrayutilities / pymca 
# SJL - 20150114

#import xrayutilities as xu #obsolete now use pymca
#import silx.io.specfilewrapper as sfwr
from builtins import zip
from builtins import map
from builtins import input
from builtins import range
from past.utils import old_div
from PyMca5 import specfilewrapper as sfwr
from PyMca5.PyMca import EdfFile
from PyMca5.PyMca import ArraySave

import h5py
import numpy as np
import os, sys
import pylab as pl


######################
# complete functions #
######################
# print_hdf5_file_structure(file_name)
# print_hdf5_item_structure(g, offset='    ')
# create_groups_skeleton_h5(h5file,scan_no)
# add_user_attrs(h5file)
# create_hdf5(specfile = '.spec',specdir = '',imagedir = '',scan_nos = [0],output_fn='raw_data_',savetiff = False,tiffdir='',mask=False)
# get_scan_images(h5fn,scan_no)
#   cleanup() - related to get scan images
# make_ff_h5(ff_class, fn='ff.h5', save_raw_ims = False) - dependency - flatfield.py
# read_ff_h5 - dependency - flatfield.py
######################




def print_hdf5_file_structure(file_name):
    """Prints the HDF5 file structure"""
    file = h5py.File(file_name, 'r') # open read-only
    item = file #["/Configure:0000/Run:0000"]
    print_hdf5_item_structure(item)
    file.close()
 
def print_hdf5_item_structure(g, offset='    ') :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    
    if   isinstance(g,h5py.File) :
        print((g.file, '(File)', g.name))
 
    elif isinstance(g,h5py.Dataset) :
        print(('(Dataset)', g.name, '    len =', g.shape)) #, g.dtype
 
    elif isinstance(g,h5py.Group) :
        print(('(Group)', g.name))
 
    else :
        print(('WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name))
        sys.exit ( "EXECUTION IS TERMINATED" )
 
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        for key,val in list(dict(g).items()) :
            subg = val
            #print(offset, key, end=' ') #,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ')

def create_groups_skeleton_h5(h5file,scan_no):
    # A bit of Nexus formatting 
    # Create some groups (think of groups as a dictionary (of datasets or other groups))
    h5file.create_group('/scan_%.4i'%scan_no)
    h5file.create_group('/scan_%.4i/vars'%scan_no)
    # scan parameters line from spec 'ascan ....', exposure / timestamp
    h5file.create_group('/scan_%.4i/data'%scan_no)
    # image_data, spec_data - EVERYTHING SHOULD LINK TO THIS FOR MOTORS - use pymca spec reader or xu ? or other?
    h5file.create_group('/scan_%.4i/instrument'%scan_no)
    h5file.create_group('/scan_%.4i/instrument/detector'%scan_no)
    # name, flatfield, dimensions, pixel sizes, central pixel, timestamp, threshold
    h5file.create_group('/scan_%.4i/instrument/diffractometer'%scan_no)
    # horizontal
    h5file.create_group('/scan_%.4i/instrument/diffractometer/motors'%scan_no)
    # hexapod / xv / alp/oh del gamma nu ov  
    h5file.create_group('/scan_%.4i/instrument/beam_conditioner'%scan_no)
    # filters / transm / I_zero / scalers / slit sizes
    h5file.create_group('/scan_%.4i/instrument/optics'%scan_no)
    # all the set up params X1/X2/MI1/MI2
    h5file.create_group('/scan_%.4i/instrument/undulator'%scan_no)
    # energy / gap / lambda /current 
    h5file.create_group('/scan_%.4i/sample'%scan_no)
    # UB matrix + fit params / miscuts / unit cell / name / environment
    # Gas needs an update of sdStartup maybe a new samples option which updates all of this information
    
def add_user_attrs(h5file):
    # name/ address/ email/ telephone_no 
    h5file.create_group('/user')
    h5file['user'].attrs.create('expt_id',eval(input('input expt_id: ')))
    h5file['user'].attrs.create('name',eval(input('input name: ')))
    h5file['user'].attrs.create('email',eval(input('input email: ')))
    h5file['user'].attrs.create('comments',eval(input('any other comments: ')))
    
def add_setup_specific_attrs(h5file):
    # beamline layout
    # instruments in WBM DCM/CC 
    # beam conditioner
    pass
    
def add_links_for_all_useful_information():
    pass

def decrypt_specfile_header():
    # depends on what spec reader we go for
    pass
    
def create_hdf5(specfile='.spec', image_prefix='', image_suffix='.edf.gz',
                specdir='', imagedir='', scan_nos=[0], output_fn='raw_data_',
                savetiff=False, tiffdir='', mask=False, im_dim=[516,516],
                sigfig="%05d", detector='mpx4'):
    # make an individual file for each spec scan
    # this, is because of speed when it comes to any k-map files or large mesh type scans
    # the multiple read and single write is not mainstream yet so I have not used it
    basepath = os.getcwd()
    det_im_id = detector+'inr'
    
    if savetiff:
        import tifffile
        if os.path.isdir(tiffdir):
            tiffdir = os.path.abspath(tiffdir)
            print(("path exists for tiff files: ", tiffdir))
        else:
            try:
                os.mkdir(tiffdir)
                tiffdir = os.path.abspath(tiffdir)
            except:
                print((tiffdir, 'cannot be created'))
            
    #s = xu.io.SPECFile(specfile, path=specdir)
    sf = sfwr.Specfile(specdir+specfile)
    
    #print tiffdir
    if mask:
        ID01_detmask = h5py.File(mask,'r')
    
    h5file = basepath+'/'+output_fn+".h5"
    
    if os.path.isfile(h5file):
        ID01_h5 = h5py.File(h5file,'a')#'w'  
        print("file exists.. append")
    else:
        ID01_h5 = h5py.File(h5file,'w')#'w'
        print("creating new file: "+h5file)
    for scan in scan_nos:
        try:
            create_groups_skeleton_h5(ID01_h5,scan)
            # read spec scan
            
            # xu
            #tmp = s.scan_list[scan]
            #tmp.ReadData()
            #data = tmp.data
            #scanlength = int(data[det_im_id][-1]-data[det_im_id][0])
            #ID01_h5['/scan_%.4i/data/spec_data'%scan]=data
            # end xu
            
            # sfwr
            tmp = sf[scan-1]
            #ID01_h5['/scan_%.4i/data/spec_motors'%scan]=dict(zip(sf.allmotors(),tmp.allmotorpos()))
            ID01_h5.create_group('/scan_%.4i/data/spec_motors'%scan)
            for a,b in zip(sf.allmotors(),tmp.allmotorpos()):
                ID01_h5['/scan_%.4i/data/spec_motors'%scan].attrs.create(a,b)
            #ID01_h5['/scan_%.4i/data/spec_scan'%scan]=dict(zip(tmp.alllabels(),tmp.data()))
            #np.core.records.fromarrays((sf[scan]).data(),names = (sf[scan]).alllabels()) # works but doesn't catch same heading error
            ID01_h5['/scan_%.4i/data/spec_scan'%scan] = tmp.data()
            ID01_h5['/scan_%.4i/data/spec_scan'%scan].attrs.create('header',tmp.alllabels())
            scanlength = tmp.data().shape[1]
            #tmp.data()[(tmp.alllabels()).index(det_im_id)]
            
            im_nos=[tmp.data()[(tmp.alllabels()).index(det_im_id)][0],tmp.data()[(tmp.alllabels()).index(det_im_id)][-1]]
            #im_nos = [0,37]
            if (im_nos[0]==im_nos[1]):
                print('no images... were they saved? check mpx4inr in your specfile')
            #print im_nos
            #print scanlength
            # end sfwr
            
            #image_data = ID01_h5.create_dataset('/scan_%.4i/data/image_data'%scan, np.array([scanlength,im_dim[0],im_dim[1]]),compression="gzip", compression_opts=9)#,chunks=(32,32,32))
            image_data = ID01_h5.create_dataset('/scan_%.4i/data/image_data'%scan,
                                                (1,im_dim[0],im_dim[1]),
                                                maxshape=(None,im_dim[0],im_dim[1]),
                                                compression="gzip",
                                                compression_opts=9)

            #for i,im_no in enumerate(np.arange(data[det_im_id][0],data[det_im_id][-1],1)):#data[det_im_id][-1] # xu line
            for i,im_no in enumerate(np.arange(im_nos[0],im_nos[1]+1,1)):
                ID01_h5['scan_%.4i/data/image_data'%scan].resize((i+1,im_dim[0],im_dim[1]))
                #print i,' / ', scanlength
                sys.stdout.write('\r')
                sys.stdout.write("scan %i - progress: %.2f%%"%(scan,(old_div(float(i),float(scanlength)))*100))
                sys.stdout.flush()
                
                efile = imagedir+image_prefix+sigfig%im_no+image_suffix
                #print efile
                #print image_suffix,imagedir,image_prefix
                if image_suffix =='.edf':
                    e = EdfFile.EdfFile(efile)#, path=specdir)
                if image_suffix =='.edf.gz':
                    e = EdfFile.EdfFile(efile)#, path=specdir)\
                else:
                    print(('oh dear what can the matter be? suffix is wrong exist?',efile))
                #e.ReadData()
                if mask:
                    ID01_h5['scan_%.4i/data/image_data'%scan][i,:,:] = e.GetData(0)*ID01_detmask['mask']
                    #pl.imshow(e.data*ID01_detmask['mask'])
                    #pl.show()
                    print("applied mask")
                    if savetiff:
                        tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,np.float32(e.GetData(0)*ID01_detmask['mask'])) # imageJ doesn't support float64
                else:
                    ID01_h5['scan_%.4i/data/image_data'%scan][i,:,:] = e.GetData(0)
                    if savetiff:
                        tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,e.GetData(0))    
                """
                # use xrayutilities
                e = xu.io.edf.EDFFile(efile)#, path=specdir)
                #e.ReadData()
                if mask:
                    ID01_h5['scan_%.4i/data/image_data'%scan][:,:,i] = e.data*ID01_detmask['mask']
                    #pl.imshow(e.data*ID01_detmask['mask'])
                    #pl.show()
                    print "applied mask"
                    if savetiff:
                        tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,np.float32(e.data*ID01_detmask['mask'])) # imageJ doesn't support float64
                else:
                    ID01_h5['scan_%.4i/data/image_data'%scan][:,:,i] = e.data
                    if savetiff:
                        tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,e.data) 
                """    
        except ValueError:
            print('scan %i exists moving on...'%scan)
    ID01_h5.close()
    if mask:
        ID01_detmask.close()

def generate_h5_10k(dir='./',prefix = 'data_',suffix = '.edf.gz',format = '%.4i', n_file = 1000, chunking = False, max_fn = False): #(32,32,50)
    # segregate a directory of images into a directory of h5 files with 10000 images in each file
    # painful to open a large dataset and then load it in frame by frame -
    import re
    files = os.listdir(dir)
    list_data = []
    list_no=[]
    for file in files:
        if file.startswith(prefix) and file.endswith(suffix):
            list_data.append(file)
            list_no.append(int(re.findall("\d+",list_data[-1])[0]))
    list_data.sort()
    #print max(list_no)
    if not max_fn:
        max_fn = max(list_no)
    #print max_fn
    for i in np.arange(0,max_fn,n_file):
        h5fn = prefix+'%.5i_%.5i'%(i,i+n_file)+'.h5'
        ID01_h5 = h5py.File(h5fn,'w')
        """
        if chunking:
            image_data = ID01_h5.create_dataset('/image_data', np.array([516,516,n_file]),compression="gzip", compression_opts=9,chunks=chunking)
        else:
            image_data = ID01_h5.create_dataset('/image_data', np.array([516,516,n_file]),compression="gzip", compression_opts=9)#,chunks=(32,32,50))
        """
        for j in range(n_file):
            sys.stdout.write('\r')
            sys.stdout.write("progress: %.2f%%"%((old_div(float(j),float(n_file)))*100))
            sys.stdout.flush()
            try:
                number=j+i
                efile = dir+prefix+format%number+suffix
                #print prefix+format%number+suffix
                e = EdfFile.EdfFile(efile)
                #ID01_h5['image_data'][:,:,number] = e.GetData(0)
                #ID01_h5['image_%i'%number] = e.GetData(0)
                ID01_h5.create_dataset('image_%i'%number, data=e.GetData(0),compression="gzip", compression_opts=9)
                #ID01_h5.close('image_%i'%number)
            except:
                print((efile, 'doesnt exist'))

        ID01_h5.close()
        
def create_hdf5_mnecounter(specfile = '.spec',image_prefix = '',specdir = '',imagedir = '',scan_nos = [0],output_fn='raw_data_',savetiff = False,tiffdir='',mask=False, im_dim=[516,516], counter = 'mpx4inr'):
    # make an individual file for each spec scan
    # this is because of speed when it comes to any k-map files or large mesh type scans
    # the multiple read and single write is not mainstream yet so I have not used it
    basepath = os.getcwd()

    if savetiff:
        import tifffile
        if os.path.isdir(tiffdir):
            tiffdir = os.path.abspath(tiffdir)
            print(("path exists for tiff files: ", tiffdir))
        else:
            try:
                os.mkdir(tiffdir)
                tiffdir = os.path.abspath(tiffdir)
            except:
                print((tiffdir, 'cannot be created'))

    #s = xu.io.SPECFile(specfile, path=specdir)
    sf = sfwr.Specfile(specdir+specfile)

    #print tiffdir
    if mask:
        ID01_detmask = h5py.File(mask,'r')

    for scan in scan_nos:
        h5file = basepath+'/'+output_fn+"%i.h5"%(scan)
        ID01_h5 = h5py.File(h5file,'w')#'w'
        create_groups_skeleton_h5(ID01_h5,scan)
        # read spec scan

        # xu
        #tmp = s.scan_list[scan]
        #tmp.ReadData()
        #data = tmp.data
        #scanlength = int(data[counter][-1]-data[counter][0])
        #ID01_h5['/scan_%.4i/data/spec_data'%scan]=data
        # end xu

        # sfwr
        tmp = sf[scan-1]
        #ID01_h5['/scan_%.4i/data/spec_motors'%scan]=dict(zip(sf.allmotors(),tmp.allmotorpos()))
        ID01_h5.create_group('/scan_%.4i/data/spec_motors'%scan)
        for a,b in zip(sf.allmotors(),tmp.allmotorpos()):
            ID01_h5['/scan_%.4i/data/spec_motors'%scan].attrs.create(a,b)
        #ID01_h5['/scan_%.4i/data/spec_scan'%scan]=dict(zip(tmp.alllabels(),tmp.data()))
        #np.core.records.fromarrays((sf[scan]).data(),names = (sf[scan]).alllabels()) # works but doesn't catch same heading error
        ID01_h5['/scan_%.4i/data/spec_scan'%scan] = tmp.data()
        ID01_h5['/scan_%.4i/data/spec_scan'%scan].attrs.create('header',tmp.alllabels())
        scanlength = tmp.data().shape[1]
        tmp.data()[(tmp.alllabels()).index(counter)]
        im_nos=[tmp.data()[(tmp.alllabels()).index(counter)][0],tmp.data()[(tmp.alllabels()).index(counter)][-1]]
        # end sfwr

        image_data = ID01_h5.create_dataset('/scan_%.4i/data/image_data'%scan,(1,im_dim[0],im_dim[1]),maxshape=(None,im_dim[0],im_dim[1]),compression="gzip", compression_opts=9)

        #for i,im_no in enumerate(np.arange(data['mpx4inr'][0],data['mpx4inr'][-1],1)):#data['mpx4inr'][-1] # xu line
        for i,im_no in enumerate(np.arange(im_nos[0],im_nos[1],1)):
            ID01_h5['scan_%.4i/data/image_data'%scan].resize((i+1,im_dim[0],im_dim[1]))
            #print i,' / ', scanlength
            sys.stdout.write('\r')
            sys.stdout.write("progress: %.2f%%"%((old_div(float(i),float(scanlength)))*100))
            sys.stdout.flush()
            efile = imagedir+image_prefix+"_%05d.edf"%im_no
            e = EdfFile.EdfFile(efile)#, path=specdir)
            #e.ReadData()
            if mask:
                ID01_h5['scan_%.4i/data/image_data'%scan][i,:,:] = e.GetData(0)*ID01_detmask['mask']
                #pl.imshow(e.data*ID01_detmask['mask'])
                #pl.show()
                print("applied mask")
                if savetiff:
                    tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,np.float32(e.GetData(0)*ID01_detmask['mask'])) # imageJ doesn't support float64
            else:
                ID01_h5['scan_%.4i/data/image_data'%scan][i,:,:] = e.GetData(0)
                if savetiff:
                    tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,e.GetData(0))
            """
            # use xrayutilities
            e = xu.io.edf.EDFFile(efile)#, path=specdir)
            #e.ReadData()
            if mask:
                ID01_h5['scan_%.4i/data/image_data'%scan][i,:,:] = e.data*ID01_detmask['mask']
                #pl.imshow(e.data*ID01_detmask['mask'])
                #pl.show()
                print "applied mask"
                if savetiff:
                    tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,np.float32(e.data*ID01_detmask['mask'])) # imageJ doesn't support float64
            else:
                ID01_h5['scan_%.4i/data/image_data'%scan][i,:,:] = e.data
                if savetiff:
                    tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,e.data)
            """
        ID01_h5.close()
        if mask:
            ID01_detmask.close()


def create_hdf5_kmap(specfile = '.spec',image_prefix = '',specdir = '',
                     imagedir = '',scan_nos = [0],output_fn='raw_data_',
                     savetiff = False,tiffdir='',mask=False, im_dim=[516,516]):
    # trying to add k-map compatibility sensibly - the only real solution is to generate a single file with everything in it
    # the major drawback is that this is solely dependent on write speed - which is not very fast unfortunately
    # make an individual file for each spec scan
    # this is because of speed when it comes to any k-map files or large mesh type scans
    # the multiple read and single write is not mainstream yet so I have not used it- but it won't speed up making the first hdf5 file
    basepath = os.getcwd()
    
    if savetiff:
        import tifffile
        if os.path.isdir(tiffdir):
            tiffdir = os.path.abspath(tiffdir)
            print(("path exists for tiff files: ", tiffdir))
        else:
            try:
                os.mkdir(tiffdir)
                tiffdir = os.path.abspath(tiffdir)
            except:
                print((tiffdir, 'cannot be created'))
            
    #s = xu.io.SPECFile(specfile, path=specdir)
    sf = sfwr.Specfile(specdir+specfile)
    
    #print tiffdir
    if mask:
        ID01_detmask = h5py.File(mask,'r')
        
    for scan in scan_nos:
        h5file = basepath+'/'+output_fn+"%i.h5"%(scan)
        ID01_h5 = h5py.File(h5file,'w')#'w'
        create_groups_skeleton_h5(ID01_h5,scan)
        # read spec scan
        
        # xu
        #tmp = s.scan_list[scan]
        #tmp.ReadData()
        #data = tmp.data
        #scanlength = int(data['mpx4inr'][-1]-data['mpx4inr'][0])
        #ID01_h5['/scan_%.4i/data/spec_data'%scan]=data
        # end xu
        
        # sfwr
        tmp = sf[scan-1]
        #ID01_h5['/scan_%.4i/data/spec_motors'%scan]=dict(zip(sf.allmotors(),tmp.allmotorpos()))
        ID01_h5.create_group('/scan_%.4i/data/spec_motors'%scan)
        for a,b in zip(sf.allmotors(),tmp.allmotorpos()):
            ID01_h5['/scan_%.4i/data/spec_motors'%scan].attrs.create(a,b)
        #ID01_h5['/scan_%.4i/data/spec_scan'%scan]=dict(zip(tmp.alllabels(),tmp.data()))
        #np.core.records.fromarrays((sf[scan]).data(),names = (sf[scan]).alllabels()) # works but doesn't catch same heading error
        ID01_h5['/scan_%.4i/data/spec_scan'%scan] = tmp.data()
        ID01_h5['/scan_%.4i/data/spec_scan'%scan].attrs.create('header',tmp.alllabels())
        scanlength = tmp.data().shape[1]
        im_nos=[tmp.data()[(tmp.alllabels()).index('mpx4inr')][0],tmp.data()[(tmp.alllabels()).index('mpx4inr')][-1]]
        # end sfwr
        
        image_data = ID01_h5.create_dataset('/scan_%.4i/data/image_data'%scan,
                                            (1,im_dim[0],im_dim[1]),
                                            maxshape=(None,im_dim[0],im_dim[1]),
                                            compression="gzip",
                                            compression_opts=9)

        #for i,im_no in enumerate(np.arange(data['mpx4inr'][0],data['mpx4inr'][-1],1)):#data['mpx4inr'][-1] # xu line
        for i,im_no in enumerate(np.arange(im_nos[0],im_nos[1],1)):
            ID01_h5['scan_%.4i/data/image_data'%scan].resize((i+1,im_dim[0],im_dim[1]))
            #print i,' / ', scanlength
            sys.stdout.write('\r')
            sys.stdout.write("progress: %.2f%%"%((old_div(float(i),float(scanlength)))*100))
            sys.stdout.flush()
            efile = imagedir+image_prefix+"_%05d.edf"%im_no
            e = EdfFile.EdfFile(efile)#, path=specdir)
            #e.ReadData()
            if mask:
                ID01_h5['scan_%.4i/data/image_data'%scan][i,:,:] = e.GetData(0)*ID01_detmask['mask']
                #pl.imshow(e.data*ID01_detmask['mask'])
                #pl.show()
                print("applied mask")
                if savetiff:
                    tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,np.float32(e.GetData(0)*ID01_detmask['mask'])) # imageJ doesn't support float64
            else:
                ID01_h5['scan_%.4i/data/image_data'%scan][i,:,:] = e.GetData(0)
                if savetiff:
                    tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,e.GetData(0))
            """
            # use xrayutilities
            e = xu.io.edf.EDFFile(efile)#, path=specdir)
            #e.ReadData()
            if mask:
                ID01_h5['scan_%.4i/data/image_data'%scan][i,:,:] = e.data*ID01_detmask['mask']
                #pl.imshow(e.data*ID01_detmask['mask'])
                #pl.show()
                print "applied mask"
                if savetiff:
                    tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,np.float32(e.data*ID01_detmask['mask'])) # imageJ doesn't support float64
            else:
                ID01_h5['scan_%.4i/data/image_data'%scan][i,:,:] = e.data
                if savetiff:
                    tifffile.imsave(tiffdir+'/'+savetiff+'%05d.tiff'%im_no,e.data) 
            """                    
        ID01_h5.close()
        if mask:
            ID01_detmask.close()


def dump_ims2hdf(h5fn,image_fns,scan_no=0, im_dim=[516,516]):
    ID01_h5 = h5py.File(h5fn,'w')
    image_data = ID01_h5.create_dataset('/scan_%.4i/data/image_data'%scan_no,
                                        (1,im_dim[0],im_dim[1]),
                                        maxshape=(None,im_dim[0],
                                        im_dim[1]),
                                        compression="gzip",
                                        compression_opts=9)
    for i,fn in enumerate(image_fns):
        #print i
        e = EdfFile.EdfFile(fn)
        nImages = e.GetNumImages()
        for j in range(nImages):
            #print j
            #print i*nImages+j
            ID01_h5['scan_%.4i/data/image_data'%scan_no].resize((i*nImages+j+1,im_dim[0],im_dim[1]))
            ID01_h5['scan_%.4i/data/image_data'%scan_no][i*nImages+j,:,:] = e.GetData(j)
    ID01_h5.close()

                
            
def get_scan_images(h5fn,scan_no):
    ID01_h5 = h5py.File(h5fn,'r')
    data = ID01_h5['scan_%.4i/data/image_data'%scan_no][:]
    ID01_h5.close()
    return data
    
def get_motor_pos(h5fn,scan_no):
    ID01_h5 = h5py.File(h5fn,'r')
    for item in list(ID01_h5['/scan_%.4i/data/spec_motors'%scan_no].attrs.keys()):
        print((item + ":", ID01_h5['/scan_%.4i/data/spec_motors'%scan_no].attrs[item]))
        dict.append()
    ID01_h5.close()
    return 
    
def get_monitor(h5fn,scan_no,mon_name):
    ID01_h5 = h5py.File(h5fn,'r')
    specdata = ID01_h5['/scan_%.4i/data/spec_scan'%scan_no][:]
    header = ID01_h5['/scan_%.4i/data/spec_scan'%scan_no].attrs['header']
    col=header.tolist().index(mon_name)
    ID01_h5.close()
    return specdata[col]

def get_spec_scan(h5fn,scan_no):
    ID01_h5 = h5py.File(h5fn,'r')
    for item in list(ID01_h5['/scan_%.4i/data/spec_scan'%scan_no].attrs.keys()):
        print((item + ":", ID01_h5['/scan_%.4i/data/spec_scan'%scan_no].attrs[item]))
    ID01_h5.close()
    
def get_spec_scan(h5fn,scan_no):
    ID01_h5 = h5py.File(h5fn,'r')
    for item in list(ID01_h5['/scan_%.4i/data/spec_scan'%scan_no].attrs.keys()):
        print((item + ":", ID01_h5['/scan_%.4i/data/spec_scan'%scan_no].attrs[item]))
    specdata = ID01_h5['/scan_%.4i/data/spec_scan'%scan_no][:]
    header = ID01_h5['/scan_%.4i/data/spec_scan'%scan_no].attrs['header']
    ID01_h5.close()
    return specdata,header

def get_dataset(h5fn,group="/",key = ""):
    ID01_h5 = h5py.File(h5fn,'r')
    if key == "":
        print_hdf5_item_structure(ID01_h5)
    else:
        try:
            output = ID01_h5[group+key].value
            #return output
        except:
            print("File/key doesnt exist")
            print_hdf5_item_structure(ID01_h5)
            output = None
    #print "close the file"
    ID01_h5.close()
    return output

def add_dataset(h5fn,dataset,group="/",key = ""):
    print(group+key)
    ID01_h5 = h5py.File(h5fn,'a')
    if key == "":
        print_hdf5_item_structure(ID01_h5)
        print("please provide a path")
    else:
        try:
            #print group+key
            ID01_h5[group+key] = dataset
        except:
            print(("File/key doesnt exist: ",key))
            print_hdf5_item_structure(ID01_h5)
    #print "close the file"
    ID01_h5.close()  
    

def get_attribute(h5fn,attribute,group="/",key = ""):
    ID01_h5 = h5py.File(h5fn,'r')
    output=None
    if key == "":
        print_hdf5_item_structure(ID01_h5)
        print("please provide a path")
    else:
        try:
            output = ID01_h5[group].attrs[key].value
        except:
            print("File/key doesnt exist")
            print_hdf5_item_structure(ID01_h5)
    ID01_h5.close()
    #print "close the file"
    return output
    
def add_attribute(h5fn,attribute,group="/",key = ""):
    ID01_h5 = h5py.File(h5fn,'a')
    if key == "":
        print_hdf5_item_structure(ID01_h5)
        print("please provide a path")
    else:
        try:
            ID01_h5[group].attrs.create(key,attribute)
        except:
            print("File/key doesnt exist")
            print_hdf5_item_structure(ID01_h5)
    #print "close the file"
    ID01_h5.close()
    
def load_rois(fn):
    #rois = np.loadtxt(fn,dtype=[('f0', '<S10'), ('f1', '<i4'), ('f2', '<i4'),('f3', '<i4'),('f4', '<i4'),('f5', '<i4'),])
    f=open(fn,'r')
    rois = {}
    for line in f.readlines():
        if line[0] != '#':
            rois[line.split()[0]]=list(map(int,line.split()[1:6]))
        
    f.close()
    return rois
#def cleanup():
#    ID01_h5.close()
#    pass

def edfmf2hdf5(fn):
    """
    save a multiframe file to hdf5
    """

    firstFile = fn

    edf = EdfFile.EdfFile(firstFile, "r")
    nImages = edf.GetNumImages()
    data = edf.GetData(0)

    hdf5file = os.path.basename(firstFile) + ".h5"

    hdf, dataset =  ArraySave.getHDF5FileInstanceAndBuffer(hdf5file,
                                                  (nImages,
                                                   data.shape[0],
                                                   data.shape[1]),
                                                   buffername="/scan_%.4i/data/images"%0,
                                                   dtype=data.dtype,
                                                   interpretation="image",
                                                   compression='gzip',)
    #                                                   compression_opts=9)

    h5file['user'].attrs.create('expt_id',eval(input('input expt_id: ')))
    for i in range(nImages):
        print((i+1,':',nImages))
        data = edf.GetData(i)
        dataset[i, :, :] = data
    hdf.flush()
    hdf.close()


def edfsmf2hdf5(fns,out_fn=""):
    """
    save a multiframe file to hdf5
    """

    firstFile = fns[0]
    if out_fn=="":
        hdf5file = os.path.basename(firstFile) + ".h5"
    else:
        hdf5file = out_fn
    
    edf = EdfFile.EdfFile(firstFile, "r")
    nImages = edf.GetNumImages()
    data = edf.GetData(0)
    #header_attr = np.array(edf.GetHeader(0).keys())
    time_of_frame = []
    hdf, dataset_images =  ArraySave.getHDF5FileInstanceAndBuffer(hdf5file,
                                                  (nImages*len(fns),
                                                   data.shape[0],
                                                   data.shape[1]),
                                                   buffername="/scan_%.4i/data/images"%0,
                                                   dtype=data.dtype,
                                                   interpretation="image",
                                                   compression='gzip',)
    #                                                   compression_opts=9)
    
    for i,fn in enumerate(fns):
        edf = EdfFile.EdfFile(fn, "r")
        for j in range(nImages):
            #print i*nImages+j+1,':',nImages*len(fns)
            sys.stdout.write('\r')
            sys.stdout.write("progress: %.2f%%"%((old_div(float(i*nImages+j+1),float(nImages*len(fns))))*100))
            sys.stdout.flush()
            imagedata = edf.GetData(j)
            dataset_images[i*nImages+j, :, :] = imagedata
            timestamp = np.float(edf.GetHeader(j)['time_of_frame'])
            time_of_frame.append(timestamp)
    
    hdf['/scan_%.4i/data/time_of_frame'%0] = np.array(time_of_frame)
        
    hdf.flush()
    hdf.close()



def make_ff_h5(ff_class, fn='ff.h5', save_raw_ims=False):
	# check the file doesn't exist
	try:
		os.listdir(ff_class.ff_path).index(fn)
		print("File exists")
		q=eval(input("would you like to overwrite it? [y/n]"))
		if q=='y':
			ff_h5 = h5py.File(ff_class.ff_path+fn,'w') 
		else:
			new_fn = eval(input("New filename: "))
			ff_h5 = h5py.File(ff_class.ff_path+fn,'w')

	except ValueError:
		print("File doesn't exist: Creating file")
		ff_h5 = h5py.File(ff_class.ff_path+fn,'w')
	
	# add data to the file
	ff_h5.create_group('/ff')
	if save_raw_ims:
		ff_h5.create_dataset('ff/image_data', data = ff_class.data,compression='gzip', compression_opts=9)
	ff_h5.create_dataset('ff/ff', data = ff_class.ff,compression='gzip', compression_opts=9)
	ff_h5.create_dataset('ff/ff_rel_unc', data = ff_class.ff_unc,compression='gzip', compression_opts=9)
	ff_h5.create_dataset('ff/bad_pix_mask', data = ff_class.tot_mask,compression='gzip', compression_opts=9)
	for attr in list(ff_class.__dict__.keys()):
		try:
			code ="ff_h5['ff/ff'].attrs['%s'] = ff_class.%s"%(attr,attr)
			exec(code)
			print(code)
		except:
			pass
	ff_h5.close()

def read_ff_h5(ff_path, fn):

	try:
		ff_h5 = h5py.File(ff_path+fn,'r')
	except:
		print(("File does not exist: ", ff_path))
		print(os.listdir(ff_path))

	ff = ff_h5['ff/ff'].value
	ff_unc = ff_h5['ff/ff_rel_unc'].value
	ff_attrs = ff_h5['ff/ff'].attrs
	#print("FF Attributes: ", end=' ') 
	for attr in list(ff_attrs.keys()):
		print(attr) #, ' : ', ff_attrs[attr]
	ff_h5.close()
	return ff,ff_unc
# see /tests for example



# for loading in hdf5 files use direct_read dataset function it doesn't make a copy like python slicing!
# and only use the read functionality
# das ist alles!
