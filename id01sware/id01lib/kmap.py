"""
    This will contain some convenience functions for 5D-KMAP data in
    hd5 format. Maybe, they should be moved to the `xsocs` package one day
"""
import collections
import h5py
from .xrd.qconversion import kmap_get_qcoordinates as get_qcoordinates

ScanRange = collections.namedtuple("ScanRange", ['name',
                                                 'start',
                                                 'stop',
                                                 'numpoints'])

def get_scan_parameters(masterH5):
    with h5py.File(masterH5, 'r') as h5m:
        somekey = next(h5m.__iter__())
        someentry = h5m[somekey]
        scanpar = someentry["scan"]
        motors = []
        for i in range(2):
            key = "motor_%i"%i
            m = ScanRange(scanpar[key].value,
                          scanpar["%s_start"%key].value,
                          scanpar["%s_end"%key].value,
                          scanpar["%s_steps"%key].value
                )
            motors.append(m)
        Nentries = len(h5m)
    shape = (Nentries, motors[1].numpoints, motors[0].numpoints) #slow to fast
    return dict(shape=shape, motor0=motors[0], motor1=motors[1])
