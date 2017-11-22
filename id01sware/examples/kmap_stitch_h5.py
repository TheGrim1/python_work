#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------
# Description:
# Author: Carsten Richter <carsten.richter@esrf.fr>
# Created at: Tue May 23 16:52:07 CEST 2017
# Computer: rnice8-0208 
# System: Linux 3.16.0-4-amd64 on x86_64
#----------------------------------------------------------------------
"""
    This small script just stitches all Metadata KMAPs located in the working
    directory into one hdf5 file using external links.
    The order is with ascending `motor3`.
    
    Can be a template for custom stuff.
"""
from __future__ import print_function
import os
#import pylab as pl
import h5py
import glob



motor3 = "eta"
files = glob.glob("KMAP*/kmap_*.h5")

fsorted = dict()
for f in files:
    h5 = h5py.File(f, "r")
    mot3val = list(h5.values())[0]["instrument/positioners/%s"%motor3].value
    h5.close()
    fsorted[mot3val] = f



newh5 = h5py.File("kmap_all.h5", "x")
for ii, mot3val in enumerate(sorted(fsorted)):
    key = "kmap_%05i_%i.1"%(ii, ii)
    otherfile = fsorted[mot3val]
    foreignkey = os.path.split(otherfile)[1].strip(".h5")
    newh5[key] = h5py.ExternalLink(otherfile, "/%s"%foreignkey)
    print((key, otherfile, foreignkey))

#pl.plot(pl.sort(list(fsorted)), "sk")
#pl.show()