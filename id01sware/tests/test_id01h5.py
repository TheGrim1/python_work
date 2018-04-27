# -*- coding: utf-8 -*-
#
# convert a spec scan to q-space and view it
#
#
#
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import xrayutilities as xu
from id01lib import id01h5
from id01lib.plot import Normalize, DraggableColorbar
from id01lib.xrd import qconversion, geometries, detectors
import silx.io


WD = "/data/id01/inhouse/crichter/test/id01sware_testdata"

h5path = os.path.join(WD, "all_scans_in_one.h5")
h5f = h5py.File(h5path, "a")

sample = "align"

specpath = os.path.join(WD, "spec/%s.spec"%sample)
scanno = [1, 99] # This will be translated to the entry `x.1` in the hdf5 file


with id01h5.ID01File(h5path) as h5f:
    h5f.clear()
    sampleH5 = h5f.addSample(sample)
    sampleH5.importSpecFile(specpath,
                            numbers=scanno,
                            verbose=True,
                            compr_lvl=6)

    print("Trying rename...")
    sampleH5.importSpecFile(specpath,
                            numbers=dict(zip(scanno, [(i*5) for i in scanno])), # mapping for renaming (*5)
                            verbose=True,
                            compr_lvl=6,
                            overwrite=True)

    print("Trying string as scan numbers (e.g. 10.)...")
    sampleH5.importSpecFile(specpath,
                            numbers=dict(zip(["%i.1"%i for i in scanno], ["%i.1"%(11*i) for i in scanno])),
                            verbose=True,
                            compr_lvl=6,
                            overwrite=True)




