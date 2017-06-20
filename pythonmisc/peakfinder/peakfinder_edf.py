import numpy as np
import scipy.signal as sig
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
import fabio
import plotedf
import time

#getting a list of images
#paws = [p.squeeze() for p in np.vsplit(paws_data,4)]


def detect_peaks(image):
    newimage = image

    return newimage

def spline(image):
    newimage = sig.bspline(image,15)
    return newimage


def gaussian(image):
    newimage = sig.gauss_spline(image,1)

    return newimage

#applying the detection and plotting results
if __name__ == '__main__':


    oneedf = fabio.open( '/data/id13/inhouse5/THEDATA_I5_1/d_2016-06-09_inh_ihmi1224/PROCESS/SESSION2/keep/vo2_1_5_xsthscan3_maxproj.edf').data

#    plotedf.plot(oneedf)    

    start     = time.time()

    edited    = spline(oneedf)
#    edited    = np.subtract(oneedf, edited)
    tooktime  = time.time()-start

    plt.figure(figsize=(20,10), dpi=80)
    
    odatamax    = oneedf.max()
    plt.subplot(1,2,(1),title = "original, datamax = %s" % odatamax) 
    omax      = 10
    plt.imshow(oneedf,vmin=0,vmax=omax)

    edatamax    = edited.max()
    plt.subplot(1,2,(2),title = "new, took %ss, datamax = %s" % (tooktime,edatamax))
    emax      = 10
    plt.imshow(edited,vmin=0,vmax=emax)

    plt.show()
