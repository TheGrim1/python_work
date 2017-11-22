from __future__ import print_function
import numpy as np
import xrayutilities as xu
import scipy.signal


from .detectors import MaxiPix # default detector
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
                          ignore_mpx4trans=False):
    """
        ID01-specific file to convert an hdf5 formatted scan
        from angles to qspace.

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
    """


    motors = scan["instrument/positioners"]

    if energy is None:
        energy = motors["nrj"].value # keV
        print("Found energy reading: %.0feV"%(energy*1e3))

    energy *= 1000. # eV

    cen_pix = [cen_pix_vert, cen_pix_hor]
    if isinstance(detector, MaxiPix) and not ignore_mpx4trans:
            cen_pix[0] += motors["mpxz"].value/1000. / detector.pixsize[0]
            cen_pix[1] -= motors["mpxy"].value/1000. / detector.pixsize[1]

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





def kmap_get_qcoordinates(kmap_masterh5, energy, cenpix, ddistance, ignore_mpx4trans=False):
    Qx, Qy, Qz = [], [], []
    for name in sorted(kmap_masterh5):
        entry = kmap_masterh5[name]
        
        qx, qy, qz = get_qspace_vals(entry,
                          cenpix[1],
                          cenpix[0],
                          ddistance,
                          energy=energy,
                          ignore_mpx4trans=ignore_mpx4trans)
        Qx.append(qx)
        Qy.append(qy)
        Qz.append(qz)
    
    Qx = np.stack(Qx)
    Qy = np.stack(Qy)
    Qz = np.stack(Qz)
    return Qx, Qy, Qz

