#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------
# Description:
# Author: Carsten Richter <carsten.richter@esrf.fr>
# Created at: Wed Jun  7 18:28:51 CEST 2017
# Computer: rnice8-0207 
# System: Linux 3.16.0-4-amd64 on x86_64
#----------------------------------------------------------------------
import platform
PV = platform.python_version()
if PV.startswith("2."):
    from urllib2 import urlopen
elif PV.startswith("3."):
    from urllib.request import urlopen

from io import BytesIO
from PIL import Image
import numpy as np


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


def stretch_contrast(image, percentile_low=5., percentile_high=95.):
    """
        To stretch contast of an image discarding
        ((100-percentile_low) + percentile_low) percent of the data.
    """
    assert image.ndim==2, 'Wrong input shape.'
    percentile = percentile_low/100., percentile_high/100.
    isort = np.sort(image, axis=None)
    imin, imax = (np.array(percentile) * len(isort)).astype(int)
    Imin, Imax = isort[[imin,imax]]
    image = np.clip(image, Imin, Imax)
    return image


def contrast(array, model="diff"):
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






