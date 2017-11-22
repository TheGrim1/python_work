from __future__ import print_function
from __future__ import division
# Functions for aligning arrays
# Original code from Xianhui Xiao APS Sector 2
# Updated by Ross Harder
# Updated by Steven Leake 30/07/2014
from past.utils import old_div
import math as m
import numpy as np
import scipy.fftpack as sf


def GetImageRegistration(arr1, arr2, precision=1):
  assert arr1.shape==arr2.shape, "Arrays are different shape in registration"
  # 3D arrays
  if len(arr1.shape)==3:
    absarr1=np.abs(arr1)
    absarr2=np.abs(arr2)
    # compress array (sum) in each dimension, i.e. a bunch of 2D arrays
    ftarr1_0=sf.fftn(sf.fftshift(np.sum(absarr1,0))) # need fftshift for wrap around
    ftarr2_0=sf.fftn(sf.fftshift(np.sum(absarr2,0)))
    ftarr1_1=sf.fftn(sf.fftshift(np.sum(absarr1,1)))
    ftarr2_1=sf.fftn(sf.fftshift(np.sum(absarr2,1)))
    ftarr1_2=sf.fftn(sf.fftshift(np.sum(absarr1,2)))
    ftarr2_2=sf.fftn(sf.fftshift(np.sum(absarr2,2)))

    # calculate shift in each dimension, i.e. 2 estimates of shift
    result= dftregistration(ftarr1_2, ftarr2_2, usfac=precision)
    shiftx1, shifty1, =result[2:4]
    result= dftregistration(ftarr1_1, ftarr2_1, usfac=precision)
    shiftx2, shiftz1, =result[2:4]
    result= dftregistration(ftarr1_0, ftarr2_0, usfac=precision)
    shifty2, shiftz2, =result[2:4]

    # average them
    xshift = old_div((shiftx1+shiftx2),2)
    yshift = old_div((shifty1+shifty2),2)
    zshift = old_div((shiftz1+shiftz2),2)
    shift = xshift, yshift, zshift

  # 2D arrays
  elif len(arr1.shape)==2:
    ftarr1=sf.fftn(arr1)
    ftarr2=sf.fftn(arr2)
    result= dftregistration(ftarr1, ftarr2, usfac=precision)
    shift= tuple(result[2:])
  else:
    shift=None
  return shift

def idxmax(data):
   amp=np.abs(data)
   maxd=amp.max()
   idx=np.unravel_index(amp.argmax(), data.shape)
   return maxd,idx

def idxmax1(data):
   return np.where(data==data.max())


def dftregistration(buf1ft,buf2ft,usfac=100):
    """
        # function [output Greg] = dftregistration(buf1ft,buf2ft,usfac);
        # Efficient subpixel image registration by crosscorrelation. This code
        # gives the same precision as the FFT upsampled cross correlation in a
        # small fraction of the computation time and with reduced memory
        # requirements. It obtains an initial estimate of the crosscorrelation peak
        # by an FFT and then refines the shift estimation by upsampling the DFT
        # only in a small neighborhood of that estimate by means of a
        # matrix-multiply DFT. With this procedure all the image points are used to
        # compute the upsampled crosscorrelation.
        # Manuel Guizar - Dec 13, 2007

        # Portions of this code were taken from code written by Ann M. Kowalczyk
        # and James R. Fienup.
        # J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued
        # object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458
        # (1990).

        # Citation for this algorithm:
        # Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
        # "Efficient subpixel image registration algorithms," Opt. Lett. 33,
        # 156-158 (2008).

        # Inputs
        # buf1ft    Fourier transform of reference image,
        #           DC in (1,1)   [DO NOT FFTSHIFT]
        # buf2ft    Fourier transform of image to register,
        #           DC in (1,1) [DO NOT FFTSHIFT]
        # usfac     Upsampling factor (integer). Images will be registered to
        #           within 1/usfac of a pixel. For example usfac = 20 means the
        #           images will be registered within 1/20 of a pixel. (default = 1)

        # Outputs
        # output =  [error,diffphase,net_row_shift,net_col_shift]
        # error     Translation invariant normalized RMS error between f and g
        # diffphase     Global phase difference between the two images (should be
        #               zero if images are non-negative).
        # net_row_shift net_col_shift   Pixel shifts between images
        # Greg      (Optional) Fourier transform of registered version of buf2ft,
        #           the global phase difference is compensated for.
    """

    # Compute error for no pixel shift
    if usfac == 0:
        CCmax = np.sum(buf1ft*np.conj(buf2ft))
        rfzero = old_div(np.sum(abs(buf1ft)**2),buf1ft.size)
        rgzero = old_div(np.sum(abs(buf2ft)**2),buf2ft.size)
        error = 1.0 - CCmax*np.conj(CCmax)/(rgzero*rfzero)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax),np.real(CCmax))
        return error, diffphase,

    # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the
    # peak
    elif usfac == 1:
        ndim = np.shape(buf1ft)
        m = ndim[0]
        n = ndim[1]
        CC = sf.ifftn(buf1ft*np.conj(buf2ft))
        max1,loc1 = idxmax(CC)
        rloc = loc1[0]
        cloc = loc1[1]
        CCmax=CC[rloc,cloc]
        rfzero = old_div(np.sum(np.abs(buf1ft)**2),(m*n))
        rgzero = old_div(np.sum(np.abs(buf2ft)**2),(m*n))
        error = 1.0 - CCmax*np.conj(CCmax)/(rgzero*rfzero)
        error = np.sqrt(np.abs(error))
        diffphase=np.arctan2(np.imag(CCmax),np.real(CCmax))
        md2 = np.fix(old_div(m,2))
        nd2 = np.fix(old_div(n,2))
        if rloc > md2:
            row_shift = rloc - m
        else:
            row_shift = rloc

        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc

        return error,diffphase,row_shift,col_shift

    # Partial-pixel shift
    else:

        # First upsample by a factor of 2 to obtain initial estimate
        # Embed Fourier data in a 2x larger array
        ndim = np.shape(buf1ft)
        m = ndim[0]
        n = ndim[1]
        mlarge=m*2
        nlarge=n*2
        CC=np.zeros([mlarge,nlarge],dtype=np.complex128)

        CC[(m-np.fix(old_div(m,2))):(m+1+np.fix(old_div((m-1),2))),(n-np.fix(old_div(n,2))):(n+1+np.fix(old_div((n-1),2)))] = (sf.fftshift(buf1ft)*np.conj(sf.fftshift(buf2ft)))[:,:]

        # Compute crosscorrelation and locate the peak
        CC = sf.ifftn(sf.ifftshift(CC)) # Calculate cross-correlation
        max1,loc1 = idxmax(np.abs(CC))
        rloc = loc1[0]
        cloc = loc1[1]
        CCmax = CC[rloc,cloc]

        # Obtain shift in original pixel grid from the position of the
        # crosscorrelation peak
        ndim = np.shape(CC)
        m = ndim[0]
        n = ndim[1]

        md2 = np.fix(old_div(m,2))
        nd2 = np.fix(old_div(n,2))
        if rloc > md2:
            row_shift = rloc - m
        else:
            row_shift = rloc

        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc

        row_shift=old_div(row_shift,2)
        col_shift=old_div(col_shift,2)

        # If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2:
            ### DFT computation ###
            # Initial shift estimate in upsampled grid
            row_shift = 1.*np.round(row_shift*usfac)/usfac;
            col_shift = 1.*np.round(col_shift*usfac)/usfac;
            dftshift = np.fix(old_div(np.ceil(usfac*1.5),2)); ## Center of output array at dftshift+1
            # Matrix multiply DFT around the current shift estimate
            CC = old_div(np.conj(dftups(buf2ft*np.conj(buf1ft),np.ceil(usfac*1.5),np.ceil(usfac*1.5),usfac,\
dftshift-row_shift*usfac,dftshift-col_shift*usfac)),(md2*nd2*usfac**2))
            # Locate maximum and map back to original pixel grid
            max1,loc1 = idxmax(np.abs(CC))
            rloc = loc1[0]
            cloc = loc1[1]

            CCmax = CC[rloc,cloc]
            rg00 = old_div(dftups(buf1ft*np.conj(buf1ft),1,1,usfac),(md2*nd2*usfac**2))
            rf00 = old_div(dftups(buf2ft*np.conj(buf2ft),1,1,usfac),(md2*nd2*usfac**2))
            rloc = rloc - dftshift
            cloc = cloc - dftshift
            row_shift = 1.*row_shift + 1.*rloc/usfac
            col_shift = 1.*col_shift + 1.*cloc/usfac

        # If upsampling = 2, no additional pixel shift refinement
        else:
            rg00 = np.sum(buf1ft*np.conj(buf1ft))/m/n;
            rf00 = np.sum(buf2ft*np.conj(buf2ft))/m/n;

        error = 1.0 - CCmax*np.conj(CCmax)/(rg00*rf00);
        error = np.sqrt(np.abs(error));
        diffphase = np.arctan2(np.imag(CCmax),np.real(CCmax));
        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        if md2 == 1:
            row_shift = 0

        if nd2 == 1:
            col_shift = 0;

        # Compute registered version of buf2ft
##        if (usfac > 0):
##            ndim = np.shape(buf2ft)
##            nr = ndim[0]
##            nc = ndim[1]
##            Nr = sf.ifftshift(np.arange(-np.fix(1.*nr/2),np.ceil(1.*nr/2)))
##            Nc = sf.ifftshift(np.arange(-np.fix(1.*nc/2),np.ceil(1.*nc/2)))
##            Nc,Nr = np.meshgrid(Nc,Nr)
##            Greg = buf2ft*np.exp(1j*2*np.pi*(-1.*row_shift*Nr/nr-1.*col_shift*Nc/nc))
##            Greg = Greg*np.exp(1j*diffphase)
##        elif (nargout > 1)&(usfac == 0):
##            Greg = np.dot(buf2ft,exp(1j*diffphase))

        # return error,diffphase,row_shift,col_shift,Greg
        return error,diffphase,row_shift,col_shift


def dftups(inp,nor,noc,usfac=1,roff=0,coff=0):
    """
        # function out=dftups(in,nor,noc,usfac,roff,coff);
        # Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
        # a small region.
        # usfac         Upsampling factor (default usfac = 1)
        # [nor,noc]     Number of pixels in the output upsampled DFT, in
        #               units of upsampled pixels (default = size(in))
        # roff, coff    Row and column offsets, allow to shift the output array to
        #               a region of interest on the DFT (default = 0)
        # Recieves DC in upper left corner, image center must be in (1,1)
        # Manuel Guizar - Dec 13, 2007
        # Modified from dftus, by J.R. Fienup 7/31/06

        # This code is intended to provide the same result as if the following
        # operations were performed
        #   - Embed the array "in" in an array that is usfac times larger in each
        #     dimension. ifftshift to bring the center of the image to (1,1).
        #   - Take the FFT of the larger array
        #   - Extract an [nor, noc] region of the result. Starting with the
        #     [roff+1 coff+1] element.

        # It achieves this result by computing the DFT in the output array without
        # the need to zeropad. Much faster and memory efficient than the
        # zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]
    """

    ndim = np.shape(inp)
    nr = ndim[0]
    nc = ndim[1]

    # Compute kernels and obtain DFT by matrix products
    a = np.zeros([nc,1])
    a[:,0] = ((sf.ifftshift(np.arange(nc)))-np.floor(1.*nc/2))[:]
    b = np.zeros([1,noc])
    b[0,:] = (np.arange(noc)-coff)[:]
    kernc = np.exp((-1j*2*np.pi/(nc*usfac))*np.dot(a,b))

    a = np.zeros([nor,1])
    a[:,0] = (np.arange(nor)-roff)[:]
    b = np.zeros([1,nr])
    b[0,:] = (sf.ifftshift(np.arange(nr))-np.floor(1.*nr/2))[:]
    kernr = np.exp((-1j*2*np.pi/(nr*usfac))*np.dot(a,b))

    return np.dot(np.dot(kernr,inp),kernc)


if __name__=="__main__":
  import Tools as t
  import pylab as pl

  a=np.zeros((64,64,64),dtype="Complex64")
  a[32-10:32+10,32-10:32+10,32-10:32+10]=1

  a=pl.imread('ss.png')
  a=a[:,:,0]

  b=a.copy()

  #b=np.roll(b,1,1)
  #b=np.roll(b,-10,0)
  b=t.Shift(b, (-4.665,-3.234))

  #b=t.Shift(b, (-4.665,-3.234,-1.36))

  print(GetImageRegistration(a,b,precision=1000))



