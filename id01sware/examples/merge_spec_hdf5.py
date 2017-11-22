#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------
# Description:
# Author: Carsten Richter <carsten.richter@esrf.fr>
# Created at: Thu Jun 29 14:42:38 CEST 2017
# Computer: lid01gpu1 
# System: Linux 3.16.0-4-amd64 on x86_64
#
# Copyright (c) 2017 Carsten Richter  All rights reserved.
#----------------------------------------------------------------------
import os
import glob
from id01lib import id01h5 # all the merging is there
from silx.io.spech5 import SpecH5
from datetime import datetime
samples = [
            "align",
            "desy8_hzo"
          ]

generic_path = "../spec/%s*.spec"

with id01h5.ID01File("all_scans_in_one.h5") as h5f:
    for name in samples:
        sample = h5f.addSample(name)
        flist = glob.glob(generic_path%name)
        
        for path in flist:
            sample.importSpecFile(path, verbose=True, compr_lvl=6)

    h5f.flush()






