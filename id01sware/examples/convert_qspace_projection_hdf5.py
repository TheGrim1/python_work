#!/usr/bin/env python
import h5py
from id01lib.xrd import qconversion
import pylab as pl
# Open file and some definitions
h5f = h5py.File('/data/id01/inhouse/UPBLcomm/HC3254/hdf5/si_standard.h5','r')
sample = "si_standard"
scans = [1,2,3]
bins = -1 # default

cen_pix_x = 214.382
cen_pix_y = 285.081
nrj=9
det_distance = 560.706
monitor="exp1"

for scan_no in scans:
    # reconstruct
    q, gint = qconversion.scan_to_qspace_h5(
    h5f[sample]['{0}.1'.format(scan_no)],
    cen_pix_x,
    cen_pix_y,
    det_distance,
    nbins=bins,
    medfilter=False,
    projection="radial",
    monitor=monitor)
    
    pl.plot(q, gint)

h5f.close()
pl.show()