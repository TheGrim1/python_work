#TODO - multiple detector output - example - not clear how to do it
#TODO - ask the question for overwrite rather than skipping i.e if a mistake is made 
#TODO - delete a scan from an hdf5 file cleanly - use 'del' function
#TODO - varying count rates between images


# make a flatfield of Eiger data
# generate the data
from id01lib import id01h5
import glob

sample = 'align' # name of top level h5 entry
h5file = 'align.h5' # output file
imgdir = None#'/data/visitor/hc3211/id01/mpx/e17089/'

#speclist = glob.glob(specdir+'e16014.spec')
specfile = '/data/visitor/hc3198/id01/spec/%s.spec'%sample # source file
scanno = ("16.1", "17.1", "18.1", "19.1", "20.1", "21.1", "22.1", "23.1", "24.1", "25.1", "26.1", "27.1") # None for all, must be tuple i.e. ("1.1",) for single scanno

with id01h5.ID01File(h5file,'a') as h5f: # the safe way
    s = h5f.addSample(sample)
    s.importSpecFile(specfile,
                     numbers=scanno,
                     verbose=True,
                     imgroot=imgdir,
                     overwrite=False, # skip if scan exists
                     compr_lvl=6)


from id01lib.flatfield import Flatfieldv2
import h5py as h5
import sys
import numpy as np

data=h5.File('align.h5','r')
#mask=h5.File('/data/id01/inhouse/leake/beamReconstructions/EigerMask.h5','r')['/image_data'].value

#create a flatfield
ff_path = ''

# remove bad frames, eiger got some bad packets
remove=[3,25,26,44,54,55,60,66,73,76,77,90,91,92,94,96,102,116,123,131,132,136,137,142,152,157,165,180,181,189,192]

ff_data0 = np.r_[data['/align/19.1/measurement/image_0/data'][:,:,:],\
      data['/align/21.1/measurement/image_0/data'][:,:,:],\
      data['/align/23.1/measurement/image_0/data'][:,:,:],\
      data['/align/25.1/measurement/image_0/data'][:,:,:],\
      data['/align/27.1/measurement/image_0/data'][:,:,:]]
    
ff_data0=np.delete(ff_data0,remove,axis=0)
#ff_monitor = data[u'align/16.1/measurement/exp1'][:]
ff_data0=np.delete(ff_data0,np.arange(0,42,1),axis=0)
ff_data0=np.delete(ff_data0,[54,87,88,89],axis=0)

ff_data0[ff_data0>=1000000]=0
ff_path=''

ff_init = Flatfieldv2(ff_data0, ff_path,detector='eiger2M', auto = False) #,mask=mask,auto = False)
#ff_data1 = h5.get_scan_images(h5fn,2)[:]
#ff_data2 = h5.get_scan_images(h5fn,3)[:]
#ff_data3 = h5.get_scan_images(h5fn,4)[:]
#ff_init = Flatfield(np.r_[ff_data0,ff_data1,ff_data2,ff_data3], ff_path,auto = True)

#apply a monitor
#ff_init.apply_monitor2data(ff_monitor)

ff_init.calc_I_bar_sigma()

# catch some hot pixels / dead pixels
ff_init.dead_pixel_mask = np.invert((np.isnan(ff_init.I_bar)) | \
						(np.isnan(np.sum(ff_init.data_dev,axis=0))) | \
						(ff_init.I_bar>(np.median(np.round(ff_init.I_bar))*100))) 

# plot the integrated intensity in the detector as a function of image
ff_init.apply_mask2data(np.invert(ff_init.dead_pixel_mask))

ff_init.plot_int_det_cts(mask=ff_init.dead_pixel_mask)
ff_init.scale_data(mask=ff_init.dead_pixel_mask)
ff_init.plot_int_det_cts(mask=ff_init.dead_pixel_mask)

# mask_1: take only those pixels whose count rate lies within the user defined min/max count rate
#ff_init.set_I_min_max()
print("set I min/max")

ff_init.I_lims=[30,153]
ff_init.mask_1 = ((ff_init.I_bar>= ff_init.I_lims[0]) & \
                (ff_init.I_bar <= ff_init.I_lims[1])) #& \                (ff_init.dead_pixel_mask == False)

ff_init.plot_bad_pxl_mask(ff_init.mask_1,id='1')

ff_init.apply_mask2data(np.invert(ff_init.mask_1))


'''
print("plot bad pixel mask")
# mask_2 based on acceptable counting rates
ff_init.make_mask_2()
print("made mask 2")
ff_init.plot_bad_pxl_mask(ff_init.mask_2,id='2')
# mask_3 based on tolerance ~ 98%
ff_init.mask_3 = ff_init.set_tolerance()
print("set tolerance")
ff_init.plot_bad_pxl_mask(ff_init.mask_3,id='3')

ff_init.final_mask([ff_init.mask_3,ff_init.mask_2,ff_init.mask_1])
'''
ff_init.tot_mask=ff_init.dead_pixel_mask
ff_init.tot_mask=ff_init.mask_1
print("final mask")
ff_init.plot_bad_pxl_mask(ff_init.tot_mask,id='final')

# look at the standard deviation across the detector
ff_init.apply_mask2data(np.invert(ff_init.tot_mask))
ff_init.plot_SD_image(clim=[0.0,0.2])
print("plot SD image")
ff_init.gen_ff()
print("generate ff")
ff_init.plot_ff()
print("plot ff")
ff_init.plot_rnd_ff_im()
print("plot rnd ff im")
ff_init.plot_worst_pixel()
print("plot worst pixel")


#ff_init.calc_ff_ID01()
ff_init.make_ff_h5()
#ff,ff_unc = ff_init.read_ff_h5()


