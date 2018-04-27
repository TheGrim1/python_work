#TODO add real dead pixels from flatfield
#TODO refine central pixel positions based on the size of a slit.
#TODO use OTSU binariser and region bbox method

# check the detector distance from a mesh scan
# calculate a matrix of detector distances
# decide if it is a reliable value 
# get the best estimate of the value

"""
from id01lib import id01h5
"""
import glob
import h5py
import numpy as np
import scipy.ndimage as spyndim
from skimage.feature import register_translation
from scipy.ndimage.fourier import fourier_shift

sample = 'GeSn4-D16S1044P11' # name of top level h5 entry
h5file = 'det_calib.h5' # output file
imgdir = None#'/data/visitor/hc3211/id01/mpx/e17089/'

#speclist = glob.glob(specdir+'e16014.spec')
specfile = '/data/id01/inhouse/IHR/HC3302/id01/spec/%s.spec'%sample # source file
scanno = ("2.1", ) # None for all, must be tuple i.e. ("1.1",) for single scanno
"""
with id01h5.ID01File(h5file,'a') as h5f: # the safe way
    s = h5f.addSample(sample)
    s.importSpecFile(specfile,
                     numbers=scanno,
                     verbose=True,
                     imgroot=imgdir,
                     overwrite=False, # skip if scan exists
                     compr_lvl=6)

"""

def det_distance(pixperdeg,pix_size=75E-6):
    return (pixperdeg*pix_size)/np.tan(np.deg2rad(1.0))


data=h5py.File(h5file,'r')

rawdata = data['/%s/2.1/measurement/image_0/data'%sample].value
# Kill bad pixels from flatfield
rawdata[:,1640:1950,606:640]=0

rawdata_copy = rawdata.copy()


# set an ROI

p0=spyndim.measurements.center_of_mass(rawdata[0,:,:])
p1=spyndim.measurements.center_of_mass(rawdata[-1,:,:])
pxl2expand=50
roi0=[0,0]
roi1=[0,0]

if p0[0]>p1[0]:
   roi0[0]=p1[0]
   roi1[0]=p0[0]
else:
   roi0[0]=p0[0]
   roi1[0]=p1[0]

if p0[1]>p1[1]:
   roi0[1]=p1[1]
   roi1[1]=p0[1]
else:
   roi0[1]=p0[1]
   roi1[1]=p1[1]

newroi=[int(p0[0]-pxl2expand),int(p1[0]+pxl2expand),int(p0[1]-pxl2expand),int(p1[1]+pxl2expand)]
rawdata=rawdata[:,newroi[0]:newroi[1],newroi[2]:newroi[3]]

rds=rawdata.shape
rawdata=np.pad(rawdata,((0,0),(rds[1]//2,rds[1]//2),(rds[2]//2,rds[2]//2)),'constant',constant_values=(0,0))

#blur it / alternatively you could estimate the slit size and cross correlate it to find the real center
for ii in range(rawdata.shape[0]):
    rawdata[ii,:,:]=spyndim.median_filter(rawdata[ii,:,:],size=25)

# make image binary to find the actual central pixel 
rawdata1 = np.where(rawdata>rawdata.max()*0.01,1,0)#.sum(axis=0)

mot_nu = data['/%s/2.1/measurement/nu'%sample].value
mot_del = data['/%s/2.1/measurement/del'%sample].value

ref_image_no=0

deltas_nu = mot_nu-mot_nu[ref_image_no]
deltas_del = mot_del-mot_del[ref_image_no]

deltas_pxl_x = np.zeros(rawdata.shape[0])
deltas_pxl_y = np.zeros(rawdata.shape[0])

# find the shifts between all images

for ii in range(rawdata.shape[0]):
	shift, error, diffphase = register_translation(rawdata[ref_image_no], rawdata[ii,:,:], upsample_factor=100)
	print("Detected pixel offset [y,x]: [%g, %g]" % (shift[0], shift[1]))
	#offset_image2 = np.fft.ifftn(fourier_shift(np.fft.fftn(image2), shift))
	deltas_pxl_x[ii]=shift[0]
	deltas_pxl_y[ii]=shift[1]

pixperdeg_nu=deltas_pxl_y/deltas_nu*-1  # geometry added here
pixperdeg_del=deltas_pxl_x/deltas_del*-1  # geometry added here

numean=np.mean(pixperdeg_nu[np.isfinite(pixperdeg_nu)])
nustd=np.std(pixperdeg_nu[np.isfinite(pixperdeg_nu)])
delmean=np.mean(pixperdeg_del[np.isfinite(pixperdeg_del)])
delstd=np.std(pixperdeg_del[np.isfinite(pixperdeg_del)])

numean_dist=det_distance(numean)
nustd_dist=det_distance(nustd)
delmean_dist=det_distance(delmean)
delstd_dist=det_distance(delstd)

print("Using DEL: ")
print("  detector distance mean: %.5f metres"%delmean_dist)
print("      standard deviation: %.5f metres"%delstd_dist)


#estimate central pixel

toggle=True

for ii in range(rawdata.shape[0]):
    if ((mot_nu==0) & (mot_del==0))[ii]:
        COM_x,COM_y=spyndim.measurements.center_of_mass(rawdata[ii,:,:])
        toggle=False
        print("image FOUND at nu =0 del = 0 ")
        print("Central pixel @ [%.3f,%.3f]\n"%(COM_x+newroi[0]-rds[1]//2,COM_y+newroi[2]-rds[2]//2))

if toggle:
	COM_x = np.zeros(rawdata.shape[0])
	COM_y = np.zeros(rawdata.shape[0])

	for ii in range(rawdata.shape[0]):
		COM_x[ii],COM_y[ii]=spyndim.measurements.center_of_mass(rawdata[ii,:,:])

	pxl_x_mean=np.mean(COM_y+numean*mot_nu*-1)    # geometry added here
	pxl_x_std=np.std(COM_y+numean*mot_nu*-1)      # geometry added here
	pxl_y_mean=np.mean(COM_x+delmean*mot_del*-1)  # geometry added here
	pxl_y_std=np.std(COM_x+delmean*mot_del*-1)    # geometry added here
	print("image at nu =0 del = 0 not taken - estimate from all images")
	print("Central pixel @ [%.3f,%.3f]\n"%(pxl_y_mean+newroi[0]-rds[1]//2,pxl_x_mean+newroi[2]-rds[2]//2))

rawdata=rawdata_copy

import xrayutilities as xu
# start= pwidth1, pwidth2, distance, tiltazimuth, tilt, detector_rotation, outerangle_offset
#param, eps = xu.analysis.sample_align.area_detector_calib(  
#    mot_nu, mot_del, rawdata, ['z-', 'y-'], 'x+', plot=True, start=(75e-06, 75e-06, delmean_dist, 0, 0, 0, 0),
#    fix=(True, True, True, False, False, False, False), plotlog=True, debug=False)

# can be a little tedious to get it to converge 
# I keep getting a tiltazimuth which is 20 degrees off - this cannot be physical
# NB: the parameter input order does not match the output order... beware


param, eps = xu.analysis.sample_align.area_detector_calib(  
    mot_nu, mot_del, rawdata, ['z-', 'y-'], 'x+', plot=True, start=(75e-06, 75e-06, delmean_dist, 90, 0.544, 2.229, 1.4249),
    fix=(True, True, True, True, False, False, False), plotlog=True, debug=False)

"""
total time needed for fit: 312.84sec
fitted parameters: epsilon: 1.1493e-09 (12,['Problem is not full rank at solution', 'Parameter convergence']) 
param: (cch1,cch2,pwidth1,pwidth2,tiltazimuth,tilt,detrot,outerangle_offset)
param: 1377.87 651.21 7.5000e-05 7.5000e-05 0.8645 90.0 0.54 2.665 1.395
please check the resulting data (consider setting plot=True)
detector rotation axis / primary beam direction (given by user): ['z-', 'y-'] / x+
detector pixel directions / distance: z- y+ / 1
	detector initialization with: init_area('z-', 'y+', cch1=1377.87, cch2=651.21, Nch1=2164, Nch2=1030, pwidth1=7.5000e-05, pwidth2=7.5000e-05, distance=0.86452, detrot=2.665, tiltazimuth=90.0, tilt=0.542)
AND ALWAYS USE an (additional) OFFSET of 1.3952deg in the OUTER DETECTOR ANGLE!
"""



