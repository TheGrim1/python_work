"""
    This file contains functions to convert ID01 data
    into reciprocal space based on xrayutilities.

    In short:
        get_qspace_vals:
            calculates the coordinates of reciprocal space that are
            covered by a certain spec scan.

        scan_to_qspace_h5:
            based on get_qspace_vals, this one does an actual rebinning
            of the data into 3d (2d or 1d) reciprocal space (projections).

        kmap_get_qcoordinates:
            similar to get_qspace_vals, but to be applied on 5D KMAP data
            in hdf5 format.
"""
from __future__ import print_function
import numpy as np
import xrayutilities as xu
import scipy.signal
import h5py

from .detectors import MaxiPix, Eiger2M # default detector MaxiPix
#from .detectors import Andor
from .geometries import ID01psic # default geometry
#from .geometries import ID01ff # sample mounted sideways # not necessary, it's included in ID01psic


def get_qspace_vals(scan, cen_pix_hor,
                          cen_pix_vert,
                          distance,
                          detector=MaxiPix(),
                          geometry=ID01psic(),
                          energy=None,
                          ipdir=[1,0,0],
                          ndir=[0,0,1],
                          roi=None,
                          Nav=None,
                          spherical=False,
                          ignore_mpx4trans=False):
    """
        ID01-specific function to calculate qspace coordinates of a scan.

        Inputs:
            cen_pix_* : int
                the result of det_calib: cen_pix_x, cen_pix_y for maxipix

            distance : float
                sample to detector distance in meters


        Optional inputs:
            detector : `AreaDetector` class instance
                describes the detector

            geometry :  child instance of `EmptyGeometry`
                describes the diffraction geometry

            energy :  float
                the beam energy in keV. Taken from the scan header if not given

            ipdir : 3-tuple(float)
                vector referring to the inplane-direction of the sample
                (see xrayutilities.experiment)

            ndir : 3-tuple(float)
                vector parallel to the sample normal
                (see xrayutilities.experiment)
            spherical : bool
                Whether to return Q in spherical coordinates. Then returns
                    rotx (roll, second rotation)
                    roty (pitch, first rotation)
                    Qabs (length of Q vector)
    """


    motors = scan["instrument/positioners"]

    if energy is None:
        energy = motors["nrj"].value # keV
        print("Found energy reading: %.0feV"%(energy*1e3))

    energy *= 1000. # eV

    cen_pix = [cen_pix_vert, cen_pix_hor]
    if isinstance(detector, MaxiPix) and not ignore_mpx4trans:
        mpxy = motors["mpxy"].value
        mpxz = motors["mpxz"].value
        cen_pix[0] += mpxz/1000. / detector.pixsize[0]
        cen_pix[1] -= mpxy/1000. / detector.pixsize[1]
        print("Correcting mpxy=%.2f, mpxz=%.2f  ==>  cen_pix = (%.1f, %.1f)"
              %(mpxy, mpxz, cen_pix[0], cen_pix[1]))

    # convention for coordinate system:
    hxrd = xu.HXRD(ipdir, ndir, en=energy, qconv=geometry.getQconversion())

    ### make defaults of xrayutilities
    if roi is None:
        roi = [0, detector.pixnum[0], 0, detector.pixnum[1]]
    if Nav is None:
        Nav = [1,1] # should this be [3,3] if medfilter is on??

    ### definition of the detector
    hxrd.Ang2Q.init_area(detector.directions[0],
                         detector.directions[1],
                         cch1=cen_pix[0],
                         cch2=cen_pix[1],
                         Nch1=detector.pixnum[0],
                         Nch2=detector.pixnum[1],
                         pwidth1=detector.pixsize[0],
                         pwidth2=detector.pixsize[1],
                         distance=distance,
                         #chpdeg1=pixperdeg[0],
                         #chpdeg2=pixperdeg[1],
                         Nav=Nav,
                         roi=roi)


    angles = geometry.sample_rot.copy()
    angles.update(geometry.detector_rot) # order should be maintained
    maxlen = 1

    ### Get motor values from hdf5
    for angle in angles:
        if angle in geometry.usemotors:
            # must not name any variable `del` in python:
            dset = motors[angle if angle is not "delta" else "del"]
            if len(dset.shape):
                maxlen = max(maxlen, dset.shape[0])
            position = dset.value
        else:
            position = 0.
        angles[angle] = position - geometry.offsets[angle]

    for angle in angles:
        if np.isscalar(angles[angle]): # convert to array
            angles[angle] = np.ones(maxlen, dtype=float) * angles[angle]

    ### transform angles to reciprocal space coordinates for all detector pixels
    qx, qy, qz = hxrd.Ang2Q.area(*angles.values())

    if spherical:
        _x, _y, _z = qy, qz, qx # rotate coord system
        qz = np.sqrt(_x**2 + _y**2 + _z**2) # radial
        qy = np.degrees(np.arccos(_z/qz)) #rot arount qy (first)
        qx = np.degrees(np.arctan2(_y,_x)) #rot around qx (second)

    return qx, qy, qz





def scan_to_qspace_h5(scan, cen_pix_hor,
                            cen_pix_vert,
                            distance,
                            nbins=(-1,-1,-1),
                            medfilter=False,
                            detector=MaxiPix(),
                            geometry=ID01psic(),
                            energy=None,
                            monitor=None,
                            roi=None,
                            Nav=None,
                            projection=None,
                            ipdir=[1,0,0],
                            ndir=[0,0,1],
                            ignore_mpx4trans=False):
    """
        ID01-specific file to rebin an hdf5 formatted scan
        into qspace.

        Inputs:
            cen_pix_* : int
                the result of det_calib: cen_pix_x, cen_pix_y for maxipix

            distance : float
                sample to detector distance in meters


        Optional inputs:
            nbins : tuple(int)
                Number of bins used for the q-conversion.
                Two ways of input are possible:
                    - Absolute number of bins (positive integer)
                    - Multiples of the minimum bin size to combine 
                      (negative integer)
                Length Should conform to the number of dimenstions after 
                projection (see below; max 3, min 1).

                Default: (-1, -1, -1) equals to maximum number of bins

            medfilter : bool
                whether to apply a 3x3 median filter to each detector frame

            detector : `AreaDetector` class instance
                describes the detector

            geometry :  child instance of `EmptyGeometry`
                describes the diffraction geometry

            energy :  float
                the beam energy in keV. Taken from the scan header if not given

            monitor :  str
                the counter used to correct for the primary beam intensity

            roi : 4-tuple(int)
                (xmin, xmax, ymin, ymax) of the interesting region on the
                detector (see xrayutilities.experiment)

            Nav : 2-tuple(int)
                (xwidth, ywidth) of the moving average for each detector frame.
                It seems better to decrease nbins instead in the corresponding
                directions.
                (see xrayutilities.experiment)

            ipdir : 3-tuple(float)
                vector referring to the inplane-direction of the sample
                (see xrayutilities.experiment)

            ndir : 3-tuple(float)
                vector parallel to the sample normal
                (see xrayutilities.experiment)
    """

    ### make defaults of xrayutilities
    if roi is None:
        roi = [0, detector.pixnum[0], 0, detector.pixnum[1]]
    if Nav is None:
        Nav = [1,1] # should this be [3,3] if medfilter is on??

    qx, qy, qz = get_qspace_vals(scan, cen_pix_hor,
                                       cen_pix_vert,
                                       distance,
                                       detector,
                                       geometry,
                                       energy,
                                       ipdir,
                                       ndir,
                                       roi=roi,
                                       Nav=Nav,
                                       ignore_mpx4trans=ignore_mpx4trans)

    
    maxbins = []
    safemax = lambda arr: arr.max() if arr.size else 0
    for dim in (qx, qy, qz):
        maxstep = max((safemax(abs(np.diff(dim, axis=j))) for j in range(3)))
        maxbins.append(int(abs(dim.max()-dim.min())/maxstep))
    
    print("Max. number of bins: %i, %i, %i"%tuple(maxbins))

    ### get the dimensions to use
    _qdims = "xyz"
    if projection is None: # No projection
        idim = [0,1,2]
    elif len(projection)==1 and projection in _qdims:
        idim = [_qdims.index(projection)]
    elif projection=="radial":
        idim = None
    elif len(projection)==2 and projection[0] in _qdims \
                            and projection[1] in _qdims:
        idim = [_qdims.index(projection[i]) for i in (0,1)]
    else:
        raise ValueError("Invalid input for projection: %s"%str(projection))

    if not hasattr(nbins, "__iter__"):
        nbins = [nbins]
    if idim is not None: # not radial
        ### process the input for number of bins
        #nbins = map(int, nbins)
        if all([b==-1 for b in nbins]): 
            nbins = [int(maxbins[j]) for j in idim]
        elif all([b<0 for b in nbins]): 
            nbins = [int(maxbins[j]/abs(nbins[i])) for (i,j) in enumerate(idim)]
        elif all([b>0 for b in nbins]):
            pass
        else:
            raise ValueError("Invalid input for nbins: %s"%str(nbins))
        print("Reconstruct scan {0}. Qspace size: {1}.".format(scan.name, tuple(nbins)))
        toolarge = [(maxbins[j]<nbins[i]) for (i,j) in enumerate(idim)]
        if any(toolarge):
            il = toolarge.index(True)
            print("WARNING: number of bins exceeds maximum in q%s."%_qdims[idim[il]])
    else:
        if all([b==-1 for b in nbins]):
            nbins = [int(max(maxbins))]

    print("Using binning: %s"%str(nbins))


    ### preprocess images from hdf5
    image_data = scan["measurement/image_0/data"]
    #image_data = scan["instrument/detector_0/data"] # the same
    num_im = image_data.shape[0]
    # process monitor readings
    if monitor is not None:
        mon = scan["measurement/%s"%monitor].value
    else:
        mon = np.ones(num_im)
    if not (mon>0).all():
        raise ValueError("Found negative readings in monitor: %s"%monitor)

    for idx in range(num_im): # TODO: parallelize
        frame = image_data[idx]/mon[idx]
        detector.correct_image(frame) # detector specific stuff
        if medfilter: # kill some hot pixels, doesn't really work with the gaps
            frame = scipy.signal.medfilt2d(frame,[3,3])


        # moving average, data reduction
        frame = xu.blockAverage2D(frame, Nav[0], Nav[1], roi=roi)

        if not idx: # first iteration
            # create cube of empty data
            intensity = np.empty((num_im, frame.shape[0], frame.shape[1]))
            intensity[idx] = frame
        else:
            intensity[idx,:,:] = frame

    
    ### convert data to regular grid in reciprocal space
    if idim == [0,1,2]: # No projection
        gridder = xu.Gridder3D(*nbins)
        gridder(qx, qy, qz, intensity)
        return (gridder.xaxis,
                gridder.yaxis,
                gridder.zaxis,
                gridder.data)
    
    elif projection=="radial":
        qabs = np.sqrt(qx**2+qy**2+qz**2)
        gridder = xu.Gridder1D(*nbins)
        gridder(qabs, intensity)
        return (gridder.xaxis, 
                gridder.data)
    
    elif len(idim)==1:
        gridder = xu.Gridder1D(*nbins)
        gridder((qx, qy, qz)[idim[0]], intensity)
        return (gridder.xaxis,
                gridder.data)
    
    elif len(idim)==2:
        gridder = xu.Gridder2D(*nbins)
        gridder((qx, qy, qz)[idim[0]], 
                (qx, qy, qz)[idim[1]],
                intensity)
        return (gridder.xaxis,
                gridder.yaxis, 
                gridder.data)





def kmap_get_qcoordinates(kmap_masterh5, energy=None,
                                         cenpix=None,
                                         ddistance=None,
                                         detector=MaxiPix(),
                                         **kwargs):
    """
        Function to compute the cube of q-space coordinates for a 5d kmap
        scan.


        Inputs:
            kmap_masterh5: string
                path to the hdf5 kmap master file

        Optional inputs:
            energy: float
                beam energy in keV. Taken from the kmap hdf5 file if not
                given
            ddistance : float
                sample to detector distance in meters. Taken from the kmap 
                hdf5 file if not given
            cenpix : 2-tuple(int)
                Pixels of the direct beam at nu = del = 0.
                Order is (first dimension, second dimension) which is 
                usually (y, x). Taken from the kmap hdf5 file if not given

            detector : `AreaDetector` class instance
                describes the detector
        Optional key word arguments:
            ignore_mpx4trans : bool
                defines whether cenpix corresponds to mpxy = mpxz = 0
                (as in the output of `det_calib`)

            +key word arguments of `get_qspace_vals`
    """
    Qx, Qy, Qz = [], [], []
    if isinstance(kmap_masterh5, str):
        kmap_masterh5 = h5py.File(kmap_masterh5, "r")

    ignore_mpx4trans = kwargs.pop("ignore_mpx4trans", False) \
                       or not isinstance(detector, MaxiPix)

    #print(ignore_mpx4trans)

    for name in sorted(kmap_masterh5):
        entry = kmap_masterh5[name]
        
        if energy is None:
            _energy = entry["instrument/detector/beam_energy"].value/1000.
            print("found energy=%.3fkeV"%_energy, end=",  ")
        else:
            _energy = energy

        if cenpix is None:
            _cenpix = (entry["instrument/detector/center_chan_dim0"].value,
                       entry["instrument/detector/center_chan_dim1"].value)
            print("found cen pix=(%.1f, %.1f)"%_cenpix, end=",  ")
        else:
            _cenpix = cenpix

        if ddistance is None:
            pixperdeg = (entry["instrument/detector/chan_per_deg_dim0"].value,
                         entry["instrument/detector/chan_per_deg_dim1"].value)
            _ddistance = pixperdeg[0]*detector.pixsize[0]/np.tan(np.radians(1))
            print("found detector distance=%.3f"%_ddistance)
        else:
            _ddistance = ddistance
        

        qx, qy, qz = get_qspace_vals(entry,
                          _cenpix[1],
                          _cenpix[0],
                          _ddistance,
                          energy=_energy,
                          ignore_mpx4trans=ignore_mpx4trans,
                          detector=detector,
                          **kwargs)
        Qx.append(qx)
        Qy.append(qy)
        Qz.append(qz)
    
    Qx = np.stack(Qx)
    Qy = np.stack(Qy)
    Qz = np.stack(Qz)
    return Qx, Qy, Qz

