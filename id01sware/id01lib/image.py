#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------
# Description:
# Author: Carsten Richter <carsten.richter@esrf.fr>
# Created at: Wed Jun  7 18:28:51 CEST 2017
# Computer: rnice8-0207 
# System: Linux 3.16.0-4-amd64 on x86_64
#----------------------------------------------------------------------
"""
    Originally these are some functions that are used by CamView, but
    they can be useful for other routines and scripts!
"""


import platform
PV = platform.python_version()
if PV.startswith("2."):
    from urllib2 import urlopen
    from urlparse import urlparse
elif PV.startswith("3."):
    from urllib.request import urlopen
    import urllib.parse as urlparse


from io import BytesIO
from PIL import Image
import numpy as np

from . import SpecClientWrapper

_models = ["diff", "msd", "gradient"]


def url2array(url, navg=1):
    for i in range(navg):
        response = urlopen(url,timeout=4)
        im = Image.open(BytesIO(response.read()))
        if not i:
            img = np.array(im).sum(-1)
        else:
            img +=np.array(im).sum(-1)
    return img/navg



def percentile_interval(arr, low=1., high=99.):
    """
        Returns the borders of an interval having all values that are
        larger than `low`%% of values and smaller than `high`%%
        of values. This way it can be used to discard (100-high+low)%%
        of outliers.
    """
    pos = (arr.size*np.array((low,high), dtype=float)/100.).astype(int)
    vmin, vmax = np.sort(arr, axis=None)[pos]
    return vmin, vmax



def stretch_contrast(image, percentile_low=5., percentile_high=95.):
    """
        To stretch contast of an image discarding
        ((100-percentile_low) + percentile_low) percent of the data.
    """
    #assert image.ndim==2, 'Wrong input shape.'
    Imin, Imax = percentile_interval(image, percentile_low, percentile_high)
    image = np.clip(image, Imin, Imax)
    return image


def contrast(array, model="diff"):
    """
        Calculates the sharpness of a greyscale image.
        Maybe it should not be named `contrast`?
    """
    array = array.astype(float)
    #array /= array.max()
    if model=="gradient":
        ddx, ddy = np.gradient(array)
        return np.mean(ddx**2 + ddy**2)
    if model=="diff":
        return np.mean( (np.diff(array,axis=0)**2)[:,1:] + \
                        (np.diff(array,axis=1)**2)[1:,:] )
    if model=="msd":
        mean = np.mean(array)
        return np.mean((array-mean)**2)


def com(array):
    """
        center of mass of the array
    """
    array = array.astype(float)
    comx = (np.arange(array.shape[1]) * array.sum(0)).sum()/array.sum()
    comy = (np.arange(array.shape[0]) * array.sum(1)).sum()/array.sum()
    return comx, comy



def _url_validator(x):
    try:
        result = urlparse(x)
        #print(result)
        return all(map(bool, (result.scheme, result.netloc, result.path)))
    except:
        return False



class AutoFocus(object):
    """
        A class providing auto-focus functionality based on
        webcam images and an interface to spec (SpecClient)
        that allows to move a motor for focusing.

        At the moment, the optimization of focus is done
        via bounded univariate scalar function minization where
        the function returns the sharpness of the image 
        (see function `contrast`).

        As images carry noise, it would be better to use an
        optimizer for noisy functions (noisyopt?).
    """
    def __init__(self, url,
                       motor="piz",
                       limits=(50.,150.),
                       roi=None):
        self._specclient = SpecClientWrapper.SpecClientSession()
        self.url = url
        self.motor = motor
        self.limits = limits
        self.roi = roi
        self.navg = 1
        self.stretch = False
        self.contrast = "diff"

    ###### URL
    @property
    def url(self):
        return self._url
    @url.setter
    def url(self, val):
        if not _url_validator(val):
            raise ValueError("Invalid url: %s"%str(val))
        self._url = val

    ###### MOTOR
    @property
    def motor(self):
        return self._motor
    @motor.setter
    def motor(self, val):
        if not isinstance(val, (str, bytes)):
            raise ValueError("Need string as input for `motor`")
        self._motor = val
        self._motornum = self._specclient.send_sc("motor_num('%s')"%val)
    
    ###### LIMITS
    @property
    def limits(self):
        """ Limits of the auto focusing motor """
        return self._ll, self._ul
    @limits.setter
    def limits(self, val):
        if val is None:
            self._ll = -np.inf
            self._ul =  np.inf
        else:
            val = np.array(val, dtype=float, ndmin=1)
            assert len(val)==2, "Need 2 scalar values: upper and lower limit"
            self._ll = val.min()
            self._ul = val.max()
    @limits.deleter
    def limits(self):
        self._ll = -np.inf
        self._ul =  np.inf

    ###### ROI
    @property
    def roi(self):
        """
            Defines the region of interest on the picture used to evaluate the
            contrast:
                (dim0_min, dim0_max, dim1_min, dim1_max)
        """
        return (self._slice_0.start,
                self._slice_0.stop,
                self._slice_1.start,
                self._slice_1.stop)
    @roi.setter
    def roi(self, val):
        if val is None:
            self._slice_0 = slice(None,None)
            self._slice_1 = slice(None,None)
        else:
            #val = np.array(val, dtype=int)
            self._slice_0 = slice(*val[0:2])#slice(*np.sort(val[0:2]))
            self._slice_1 = slice(*val[2:4])#slice(*np.sort(val[2:4]))
    @roi.deleter
    def roi(self):
        self._slice_0 = slice(None,None)
        self._slice_1 = slice(None,None)

    ###### NAVG
    @property
    def navg(self):
        """ Number of images to average """
        return self._navg
    @navg.setter
    def navg(self, val):
        self._navg = int(val)

    ###### STRETCH
    @property
    def stretch(self):
        """
            Percentiles to stretch contrast:
                between 0 and 100, or True/False
        """
        return self._stretch
    @stretch.setter
    def stretch(self, val):
        if val is True:
            self._stretch = 5., 95.
        elif val is False:
            self._stretch = False
        else:
            val = np.array(val, dtype=float, ndmin=1)
            assert len(val)==2, ("Need 2 scalar values: upper and lower "
                                 "percentile")
            self._stretch = val.min(), val.max()

    ###### CONTRAST
    @property
    def contrast(self):
        """
            Name of model for contrast evaluation:
                One of: ["diff", "msd", "gradient"]
        """
        return self._contrast
    @contrast.setter
    def contrast(self, val):
        if not val in _models:
            #print("Problem")
            raise ValueError("Contrast model needs to be one of [%s]"
                             %", ".join(_models))
        self._contrast = val

    def focus(self, navg=None, stretch=None, contrast=None, **leastsq_kw):
        """
            do the actual focusing
        """
        if not hasattr(self, "_optimize"):
            self._optimize = __import__('scipy.optimize',
                                       globals(),
                                       locals(),
                                       ['leastsq'])

        if not navg is None:
            self.navg = navg

        if not stretch is None:
            self.stretch = stretch

        if not contrast is None:
            self.contrast = contrast

        startval = self.get_motor_pos()

#        kw = dict(full_output=True,
#                  ftol=1e-5,
#                  xtol=1e-3,
#                  maxfev=0,
#                  factor=5.)
#        kw.update(leastsq_kw)
#
#        self.result = self._optimize.leastsq(self._costfunction, startval, **kw)
        kw = dict(bracket=None,
                  bounds=self.limits,
                  method='Bounded',
                  tol=1e-3, options=None)
        kw.update(leastsq_kw)

        self.result = self._optimize.minimize_scalar(self._costfunction, **kw)
#        kw = dict(bracket=None,
#                  bounds=self.limits,
#                  method='L-BFGS-B',
#                  tol=1e-3, options=None)
#        kw.update(leastsq_kw)
#
#        self.result = self._optimize.minimize_scalar(self._costfunction)
        return self.result

    def get_motor_pos(self):
        if not self._motornum==-1:
            pos = self._specclient.get_sv("A[%i]"%self._motornum)
            return float(pos[str(self._motornum)])

    def movemotor(self, position):
        if position > self._ll and position < self._ul:
            self._specclient.send_sc("mv %s %f"%(self.motor, position))
        else:
            raise ValueError("Setpoint hits limits: %f"%position)


    def _costfunction(self, newpos=None):
        """
            return a value proportional to the inverse
            sharpness -- the function which ought to be minimized
        """
        if newpos is not None:
            self.movemotor(newpos)
        img = url2array(self.url, navg=self.navg)
        img = img[self._slice_0, self._slice_1]
        if self.stretch:
            img = stretch_contrast(img, *self.stretch)
        self._image = img
        img_contrast = contrast(img, self.contrast)
        residual = 1./img_contrast
        print("Value: %f"%residual)
        return residual


