#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------
# Description:
# Author: Carsten Richter <carsten.richter@desy.de>
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


_models = ["diff", "rsm"]

def url2array(url):
    response = urlopen(url,timeout=4)
    img = Image.open(BytesIO(response.read()))
    img = np.array(img).sum(-1)
    return img


def contrast(array, model="diff"):
    array = array.astype(float)
    if model=="diff":
        return np.sum(np.diff(array,axis=0)**2)+np.sum(np.diff(array,axis=1)**2)
    if model=="rsm":
        mean = np.mean(array)
        return np.sqrt(np.sum(np.sum((array-mean)**2))/np.prod(array.shape))

def com(array):
    """
        center of mass of the array
    """
    array = array.astype(float)
    comx = (np.arange(array.shape[1]) * array.sum(0)).sum()/array.sum()
    comy = (np.arange(array.shape[0]) * array.sum(1)).sum()/array.sum()
    return comx, comy






