#!/usr/bin/env python
#----------------------------------------------------------------------
# Description:
# Author: Carsten Richter <carsten.richter@esrf.fr>
# Created at: Do 16. Mar 16:57:08 CET 2017
# Computer: lid01gpu1.
# System: Linux 3.16.0-4-amd64 on x86_64
#----------------------------------------------------------------------
#
# Todo:
#       - autoappend to an hdf5 if the sample exists
#       - parallelization
#         (does it make sense if harddisk I/O is the biggest part of the work?)
#       - read/write of 2d slices of .edf cubes faster than whole cube?
#       - define proper nx classes + id01-specific parsing of spec file header
#       - better compatability to xsocs .h5 format
#       - more comments / doc
#
#----------------------------------------------------------------------
"""
 A layer on top of h5py and silx.io for conversion of ID01 data
 to hd5f 

 New proposed hdf5 file structure
 To be discussed...

 Adopted from silx SpecH5 combined with Edf Converter
  /
      1.1/
          title = "..."
          start_time = "..."
          instrument/
              specfile/
                  file_header = "..."
                  scan_header = "..."
              positioners/
                  motor_name = value
                  ...
              mca_0/
                  data = ...
                  calibration = ...
                  channels = ...
                  preset_time = ...
                  elapsed_time = ...
                  live_time = ...

              mca_1/
                  ...
+             detector_0/
+                 data = ...
+                 others/
+                     time = ...
+                     NumImages = ...
+                     ...
              ...
          measurement/
              colname0 = ...
              colname1 = ...
              ...
              mca_0/
                   data -> /1.1/instrument/mca_0/data
                   info -> /1.1/instrument/mca_0/
              ...
+             image_0/ --> link to detector_0
              ...
      2.1/
          ...
      3.0.kmap_00001/
          ...
      3.0.kmap_00002/
          ...
      3.0.kmap_00003/
          ...
          ...
      3.1/
          ...
"""


import os
import sys
import gzip
import h5py
import time
import itertools
import collections
import numpy as np

import silx.io
from silx.io import specfile
import silx.third_party.EdfFile as EdfFile
import silx.io.fabioh5
#from silx.io.spectoh5 import write_spec_to_h5

from id01lib.settings import edf_header_types, detector_aliases
#EdfImage = silx.io.fabioh5.fabio.edfimage.EdfImage



# Definition of the skeleton. This needs discussion and further work.
DefaultInstrument = dict()
#not compatible with silx:
#DefaultInstrument["detectors"] = dict(_descr="name, flatfield, dimensions, pixel sizes, central pixel, timestamp, threshold")
DefaultInstrument["diffractometer"] = dict(_descr="horizontal vertical")
DefaultInstrument["positioners"] = dict(_descr="hexapod, xv, alp, oh, del, gamma, nu, ov, maybe the groups from groupdump")
DefaultInstrument["beam_conditioner"] = dict(_descr="filters, transm, I_zero, scalers, slit sizes (move to optics?)")
DefaultInstrument["optics"] = dict(_descr="all the set up params X1/X2/MI1/MI2")
DefaultInstrument["source"] = dict(_descr="energy, U gap, lambda, current")
DefaultInstrument["sample_environment"] = dict(_descr="gas, temperature, etc.")

DefaultMeasurement = dict()



def isfile(path):
    try:
        return os.path.isfile(path)
    except TypeError:
        return False


def _makedefaults(g, defaults, verbose=False):
    """
        Recursive creation of a skeleton of subgroups
        from the dictionary ``defaults''
    """
    for k in defaults:
        v = defaults[k]
        if isinstance(k, str) and k.startswith("_"):
            g.attrs[k[1:]] = v
        elif isinstance(v, dict):
            newgroup = g.create_group(k)
            _makedefaults(newgroup, v)
        else:
            if verbose:
                sys.stdout.write("setting %s/%s = %s%s"%(g.name, k, str(v), os.linesep))
            g[k] = v
    return


def _len(obj):
    try:
        return len(obj)
    except:
        return 0


def _spech5_deepcopy(scan, origin="/"):
    def helper(s, obj):
        if silx._version.MINOR < 6 and silx._version.MAJOR == 0:
            s = os.path.relpath(s, origin)
        if not s:
            return
        if isinstance(obj, silx.io.spech5.SpecH5Dataset):
            if s not in scan:
                scan.create_dataset(name=s,
                                    shape=obj.shape,
                                    data=obj.value,
                                    dtype=obj.dtype,
                                    chunks=obj.chunks
                                    )
        else:
            if s not in scan:
                scan.create_group(s)
        scan[s].attrs.update(obj.attrs)
    return helper


def _read_edf_first_acq_nr(edfpath):
    """
        To find whether an .edf file is the only one of a pscan
        or if it is split up into several

        This is faster than opening the .edf file using `EdfFile`
    """
    if isinstance(edfpath, bytes):
        edfpath = edfpath.decode()
    if edfpath.endswith("edf.gz"):
        loader = gzip.open
    elif edfpath.endswith(".edf"):
        loader = open
    else:
        raise ValueError("Unsupported file: %s"%edfpath)
    with loader(edfpath) as fh:
        header = fh.read(400)
    lines = header.decode().splitlines()
    acq_frame_nb = [s for s in lines if s.startswith('acq_frame_nb')][0]
    acq_frame_nb = int(acq_frame_nb.split()[-2])
    return acq_frame_nb




class Scan(h5py.Group):
    debug = False
    def __init__(self, bind, skeleton=False):
        super(Scan, self).__init__(bind)
        if not len(self) and skeleton:
            # fill with skeleton, should be adapted
            instrument = self.create_group('instrument')
            _makedefaults(instrument, DefaultInstrument)

            measurement = self.create_group('measurement')
            _makedefaults(instrument, DefaultMeasurement)

            #header = self.create_group('others')
            # todo set default start_time?

    def addEdfFile(self, edf, detectornum=None, subdir="instrument", compr_lvl=6):
        """
            Adds an 2D or 3D EDF image to the stack of
            images or creates a new stack.
            If the detector number is not specified,
            the images will be appended to the first
            detector matching the shape.

            Accepts .edf in hdf5 format as it is provided
            after opening with silx.
        """
        if isinstance(edf, EdfFile.EdfFile):
            pass
        elif isfile(edf):
            if isinstance(edf, bytes):
                edf = edf.decode()
            edf = EdfFile.EdfFile(edf, fastedf=True)

        if not edf.NumImages:
            return

        newshape = (edf.NumImages, edf.Images[0].Dim2, edf.Images[0].Dim1)
        if self.debug:
            sys.stdout.write("Loaded image. Shape %ix%ix%i"%newshape + os.linesep)
        d = collections.defaultdict(list) # all data (images + meta)

        for i in range(edf.NumImages):
            d["data"].append(edf.GetData(i))
            for (k,v) in itertools.chain(edf.GetHeader(i).items(), edf.GetStaticHeader(i).items()):
                d[k].append(v)

        d["data"] = np.array(d["data"]) # now only metadata
        #for k in d:
        #    d[k] = np.array(d[k])
        data = d.pop("data")

        if detectornum is None:
            i = 0
            while True:
                key = "detector_%i"%i
                if not key in self[subdir]:
                    break
                if self[subdir][key]["data"].shape[1:] == newshape[1:]:
                    break
            detectornum = i
        else:
            key = "detector_%i"%detectornum

        if not key in self[subdir]:
            detector = self[subdir].create_group(key)
            detector.create_dataset("data",
                                  newshape,
                                  maxshape=(None,newshape[1],newshape[2]),
                                  compression="gzip",
                                  compression_opts=compr_lvl,
                                  dtype=data.dtype) # Todo: think about chunks - only implement for kmaps
            detector["data"][...] = data

            # rocess metadata
            others = detector.create_group("others")
            #_makedefaults(others, det["others"]) # creates non-resizable stuff
            for name, value in d.items(): # copy all metadata
                if self.debug:
                    print(name)
                if name in edf_header_types:
                    mdtype = edf_header_types[name]
                    if mdtype is bytes:
                        mdtype = h5py.special_dtype(vlen=bytes)
                    else:
                        value = np.array(value, dtype=mdtype)
                        if mdtype is np.string_:
                            mdtype = value.dtype
                else:
                    continue
                    #mdtype = bytes

                others.create_dataset(name,
                                      (len(value),),
                                      maxshape=(None,),
                                      dtype=mdtype)
                others[name][...] = value
            # silx compatability:
            #self.copy("instrument/%s"%key, "measurement/image_%i"%detectornum)
            # hard link:
            if subdir == 'instrument': # spech5 default
                self["measurement/image_%i"%detectornum] = self["%s/%s"%(subdir,key)]

        else:
            # do the resizing
            detector = self[subdir][key]
            newlen = detector["data"].shape[0] + newshape[0]
            detector["data"].resize((newlen,newshape[1],newshape[2]))
            detector["data"][-newshape[0]:] = data
            if self.debug:
                sys.stdout.write("New dataset size: "
                                 "%ix%ix%i"%detector["data"].shape
                                 + os.linesep)
            # process metadata
            others = detector["others"]
            for name, value in d.items(): # copy all metadata
                if name not in others:
                    continue
                if name in edf_header_types:
                    mdtype = edf_header_types[name]
                    #if not mdtype in (bytes, np.string_):
                    value = np.array(value, dtype=mdtype)
                newlen = others[name].shape[0] + len(value)
                others[name].resize((newlen,)) # strictly 1d
                others[name][-len(value):] = value

        return detector.name # may be useful



    def addEdfH5Image(self, Edfh5File, detectornum=None, compr_lvl=6):
        """
            Adds an 2D or 3D EDF image to the stack of
            images or creates a new stack.
            If the detector number is not specified,
            the images will be appended to the first
            detector matching the shape.

            Accepts .edf in hdf5 format as it is provided
            after opening with silx.
        """
        #print Edfh5File.keys()
        det = Edfh5File["/scan_0/instrument/detector_0"]
        data = det["data"]
        shape = data.shape

        if detectornum is None:
            i = 0
            while True:
                key = "detector_%i"%i
                if not key in self["instrument"]:
                    break
                if self["instrument"][key]["data"].shape[1:] == shape[1:]:
                    break
            detectornum = i
        else:
            key = "detector_%i"%detectornum

        if not key in self["instrument"]:
            detector = self["instrument"].create_group(key)
            detector.create_dataset("data",
                                  shape,
                                  maxshape=(None,shape[1],shape[2]),
                                  compression="gzip",
                                  compression_opts=compr_lvl,
                                  dtype=data.dtype)
            detector["data"][...] = data
            others = detector.create_group("others")
            #_makedefaults(others, det["others"]) # creates non-resizable stuff
            for name in det["others"]: # copy all metadata
                value = det["others"][name]
                others.create_dataset(name,
                                      value.shape,
                                      maxshape=(None,),
                                      dtype=value.dtype)
                others[name][...] = value

            #self.copy("instrument/%s"%key, "measurement/image_%i"%detectornum) # silx compatability
            self["measurement/image_%i"%detectornum] = self["instrument/%s"%key]

        else:
            # do the resizing
            detector = self["instrument"][key]
            newlen = detector["data"].shape[0] + shape[0]
            detector["data"].resize((newlen,shape[1],shape[2]))
            detector["data"][-shape[0]:] = data

            others = detector["others"]
            for name, value in det["others"].items(): # copy all metadata
                newlen = others[name].shape[0] + value.shape[0]
                others[name].resize((newlen,)) # strictly 1d
                others[name][-value.shape[0]:] = value


    def make_roi(self, xmin, xmax, ymin, ymax, image='image_0', store=False,
                       roinum=None):
        """
            Saves to scan["measurement/image_X_roiY"] if store==True
        """
        if isinstance(image, int):
            image = "image_%i"%image
        measurement = self.get("measurement")
        if measurement is None:
            raise ValueError("No measurement found.")
        data = self["measurement"].get(image, None)
        if data is None:
            raise ValueError("Image `%s` not found in scan."%image)
        xmin, xmax = map(int, sorted((xmin, xmax)))
        ymin, ymax = map(int, sorted((ymin, ymax)))
        roi = data[:,ymin:ymax, xmin:xmax].sum((1,2)) # y is the first image dimension
        if store:
            if roinum is not None:
                if not isinstance(roinum, int):
                    roiname = roinum
                else:
                    roiname = "%s_roi%i"%(image, roinum)
            else:
                i = 0
                while True:
                    roiname = "%s_roi%i"%(image, i)
                    if not roiname in self["measurement"]:
                        break
                    i+=1
            new = self["measurement"].create_dataset(roiname, data=roi)
            sys.stdout.write("Created dataset %s"%new.name + os.linesep)
        return roi

    def fetch_edf_spec(self, pathonly=False, verbose=True, imgroot=None,
                             **edf_kw):
        fast = self.attrs["_fast"]
        header = self["instrument/specfile/scan_header"].value
        header = header.decode().splitlines()
        fheader = self["instrument/specfile/file_header"].value
        fheader = fheader.decode().splitlines()
        specpath = os.path.realpath(self.attrs["_specfile"])
        # this is useful to get relative paths in case the complete data 
        # has been moved:
        for line in fheader:
            if line.startswith("#F"):
                orig_spec_path = os.path.realpath(line.lstrip("#F "))
                break
        orig_folder = os.path.dirname(orig_spec_path)
        #print specpath, orig_folder
        ##
        if fast:
            #impath = [s for s in header if "imageFile" in s][0]
            impath = [s for s in header if s.startswith("#C imageFile")]
            if not impath:
                return []
            impath = impath[0]
            impath = impath.split()[2:]
            impath = dict((s.strip("]").split("[") for s in impath))
            if imgroot is None:
                imgroot = os.path.realpath(impath["dir"])
                # prefer relative because data is often moved:
                imgroot = os.path.relpath(imgroot, orig_folder)
                imgroot = os.path.join(os.path.dirname(specpath), imgroot)
                imgroot = os.path.abspath(imgroot)
            generic_path = os.path.join(
                    imgroot,
                    impath["prefix"] + impath["idxFmt"] + impath["suffix"]
                    )
            idx = int(impath["nextNr"])
            #impath = generic_path%idx
            detname = "" # not used in pscans
            all_paths = FastEdfCollect(generic_path, idx)
        else:
            impath = [s for s in header if s.startswith("#ULIMA_")][0]
            detname, impath = impath.split()
            impath = os.path.realpath(impath)
            # prefer relative because data is often moved:
            impath = os.path.relpath(impath, orig_folder)
            #print impath
            impath = os.path.join(os.path.dirname(specpath), impath)
            impath = os.path.abspath(impath)
            detname = detname[7:] # discard "#ULIMA_"
            detname = detector_aliases.get(detname, detname)
            if imgroot is not None:
                impath = os.path.join(imgroot, os.path.basename(impath))

            inr = self["measurement/%sinr"%detname] #TODO: different detectors
            if not inr.len():
                return []
            startnr = int(inr[0])
            assert "_%05d"%startnr in impath, \
                 "Error: %sinr not in image path"%detname
            impath = impath.replace("_%05d"%startnr, "_%05d")
            all_paths = [impath%i for i in inr]

        all_paths = np.array(all_paths, dtype=np.string_)
        #dt = h5py.special_dtype(vlen=bytes)
        self["measurement"].create_dataset("image_files",
                                           shape=all_paths.shape,
                                           dtype=all_paths.dtype,
                                           data=all_paths)

        _i=0
        for impath in all_paths:
            impath = impath
            if not isfile(impath):
                sys.stdout.write("Warning: File not found: %s. Skipping..."%impath + os.linesep)
                continue
            if pathonly:
                if not verbose:
                    break
                sys.stdout.write("  Found path %s"%impath + os.linesep)
            else:
                try:
                    if verbose:
                        sys.stdout.write("  progress: %.2f%%: %s "%((float(_i)/float(len(all_paths)))*100,impath))
                        sys.stdout.write("\r") # carriage return only. does it work on Win?
                        sys.stdout.flush()
                    self.addEdfFile(impath, **edf_kw)
                except Exception as emsg:
                    sys.stdout.write("Could not load file: %s"%emsg)
                    sys.stdout.write(os.linesep)
            _i+=1
        sys.stdout.write(os.linesep)
        return all_paths





class Sample(h5py.Group):
    scans = []
    timefmt = "%Y-%m-%dT%H:%M:%S"
    def __init__(self, bind, description=None):
        super(Sample, self).__init__(bind)
        if not description is None:
            self.attrs["description"] = description


    def lastScan(self): # deprecated
        return (sorted(self.scans)[-1]) if self.scans else 0

    def addScan(self, number=None, *datafiles):
        """
            Trying to maintain Silx compatability.

            number : int, str
                number of the scan

            *datafiles : different kinds of data files to be processed
                         by silx
        """
        if number is None:
            number = self.lastScan() + 1
        if isinstance(number, int):
            name = '%i.1'%number
        else:
            name = number
            try:
                number = int(name.split(".")[0])
            except ValueError:
                number = -1

        if name not in self:
            with h5py._hl.base.phil:
                # new scan
                name, lcpl = self._e(name, lcpl=True)
                gid = h5py.h5g.create(self.id, name, lcpl=lcpl)
                scan = Scan(gid)
                scan.attrs["_is_a_scan"] = True
                self.scans.append(number)
        else:
            # select scan
            scan = Scan(self[name].id)

        for f in datafiles:
            if isinstance(f, bytes):
                f = f.decode()
            if isinstance(f, EdfFile.EdfFile):
                scan.addEdfFile(f) # fast !

            elif hasattr(f, "_File__fabio_image") and \
                 isinstance(f._File__fabio_image, EdfImage): #already an h5 file?
                scan.addEdfH5Image(f) # slow!

            elif isfile(f):
                if f.lower().endswith(".edf") or f.lower().endswith(".edf.gz"):
                    scan.addEdfFile(f)

        return scan

    addScanData = addScan # the same method for two purposes



    def addSpecScan(self, specscan, number=None, overwrite=False,
                          fetch_edf=True, imgroot=None, verbose=True, **edf_kw):
        if not isinstance(specscan, silx.io.spech5.SpecH5Group):
            raise ValueError("Need `silx.io.spech5.SpecH5Group` as spec scan input.")
        filename = specscan.file.filename
        #fast = filename.split("_")[-2] == "fast"
        fast = "_fast_" in os.path.basename(filename)
        root = specscan.name.split("/")[1]
        #specnumber = int(root.split(".")[0]) #+ (1 if fast else 0)
        specnumber = [int(n) for n in root.split(".")] #+ (1 if fast else 0)

        if number is None:
            if fast:
                number = int(os.path.splitext(filename)[0][-5:])
            else:
                number = specnumber


        if isinstance(number, int):
            if fast:
                name = '%i.0.kmap_%05i'%(number, specnumber[0])
                if len(specnumber)==2:
                    name = name + ".%i"%specnumber[1]
            else:
                name = '%i.1'%number
        elif _len(number)==2:
            name = '%i.%i'%tuple(number)
        else:
            name = number

        #print(specnumber)
        scan = self.addScan(name)

        if "title" in scan:
            if overwrite:
                self.pop(name)
                scan = self.addScan(name)
            else:
                sys.stdout.write("Warning: Scan %5s already exists in %s. "
                                 "Omitting. Consider the `overwrite` keyword%s"
                        %(name, self.name, os.linesep))
                return

        scan.attrs["_fast"] = fast
        scan.attrs["_specfile"] = filename

        if verbose:
            sys.stdout.write("Importing spec scan %s from %s to %s...%s"
                             %(root, filename, scan.name, os.linesep))

        #specscan.copy(specname, self, name=name) #copying does not work with spech5
        # will be replaced by spectoh5.SpecToHdf5Writer:
        specscan.visititems(_spech5_deepcopy(scan, specscan.name))

        if fetch_edf:
            scan.fetch_edf_spec(verbose=verbose, imgroot=imgroot, **edf_kw)

        return scan



    def importSpecFile(self, specfile, numbers=(), exclude=[], **addSpec_kw):
        """
            Convenience method that imports whole or parts of specfiles into the
            current `Sample` group.
            
            Inputs:
                specfile : str or silx.io.spech5.SpecH5
                    The .spec file as path or SpecH5 instance
                
                numbers : sequence of (str or int) OR dict-like
                    The subset of scans to import (all if None).
                    If the `numbers` input is a dict, it represents
                    a mapping of scan numbers/names to new, arbitrary
                    names.

                exclude : sequence
                    a subset of scans to omit.

            For additional key word arguments see `addSpecScan`.

        """
        if isfile(specfile):
            s5f = silx.io.open(specfile)
        #elif isinstance(specfile, silx.io.spech5.SpecH5):
        #    s5f = specfile
        else:
            if isinstance(specfile, str):
                raise TypeError("File not found: %s"%specfile)
            else:
                raise TypeError("Input type not supported: %s"%str(type(specfile)))

        if not numbers:
            numbers = list(s5f.keys())
        if isinstance(numbers, collections.Sequence):
            numbers = dict(zip(numbers, numbers))


        for i, number in enumerate(numbers):
            if number in exclude:
                continue

            if isinstance(number, int):
                scanno = number - 1
            else:
                scanno = number

            try:
                scan = s5f[scanno]
            except Exception as emsg:
                sys.stdout.write("Warning: Could not load scan %s - %s:%s"
                                 %(specfile,str(scanno), os.linesep))
                sys.stdout.write("    %s. Skipping..."%emsg + os.linesep)
                continue
            self.addSpecScan(scan,
                             numbers[number],
                             **addSpec_kw)


    def import_single_frames(self, image_dir, prefix="",
                                              compr_lvl=6,
                                              dest="_images_before"):
        """
            This is a way to store the frames recorded during `ct` into the
            hdf5 file. In the future there should be a reference to these
            frames in the spec file.
        """
        # first get all existing scans with startdate and image files
        times = collections.OrderedDict()
        imgfiles = []
        firstimg = dict()
        for name, scan in self.items():
            starttime = scan.get("start_time", False)
            if not starttime:
                continue
            starttime = time.mktime(time.strptime(starttime.value, self.timefmt))
            times[name] = starttime
            scanimages = scan.get('measurement/image_files', [])
            if not scanimages:
                continue
            imgfiles.extend(map(os.path.basename, scanimages))

        _alltimes = np.array(times.values())
        _allscans = list(times)

        # now process all images in image_dir that are not in imgfiles
        # and therefore not part of any scan
        ii = 0
        paths = []
        for fname in set(os.listdir(image_dir)).difference(imgfiles):
            if not fname.startswith(prefix):
                continue
            if fname.lower().endswith(".edf.gz") or \
               fname.lower().endswith(".edf"):
                path = os.path.join(image_dir, fname)
                edf = EdfFile.EdfFile(path, fastedf=True)
                try:
                    edfheader = edf.GetHeader(edf.NumImages-1)
                except: 
                    sys.stdout.write("Warning: corrupted file %s"%path + os.linesep)
                    continue
                if not "time_of_day" in edfheader:
                    nextname = "others"
                else:
                    time_last = float(edfheader['time_of_day'])
                    nexttime = _alltimes[_alltimes>time_last]
                    if not len(nexttime):
                        continue
                    nexttime = nexttime.min()
                    nextname = _allscans[np.where(_alltimes==nexttime)[0].item()]

                if nextname not in self:
                    scan = self.addScan(nextname)
                else:
                    scan = self[nextname]

                if dest not in scan:
                    destg = scan.create_group(dest)
                destg = scan[dest]

                skipit = False
                for grp in destg.values():
                    if path in grp.get("image_files", []):
                        skipit = True
                if skipit:
                    continue

                ii += 1
                sys.stdout.write("Adding single frame #%i (%s) to %s%s"
                      %(ii, fname, nextname, os.linesep))

                detname = scan.addEdfFile(edf, subdir=dest,
                                               compr_lvl=compr_lvl)

                detector = destg[detname]
                if "image_files" not in detector:
                    mdtype = h5py.special_dtype(vlen=bytes)
                    detector.create_dataset("image_files",
                                           (1,),
                                           maxshape=(None,),
                                           dtype=mdtype)
                else:
                    oldlen = detector['image_files'].shape[0]
                    detector['image_files'].resize((oldlen + 1,)) # strictly 1d
                detector['image_files'][-1] = path

            else:
                continue # todo: implement other typesprint

        return paths


    def __getitem__(self, name):
        o = super(Sample, self).__getitem__(name)
        if o.attrs.get("_is_a_scan", False):
            return Scan(o.id)
        else:
            return o



class ID01File(h5py.File):
    """
        Typical ID01 hdf5 file container
    """
    _samples = []
    def addSample(self, name, description=None):
        if name in self:
            sys.stdout.write("Warning: returning already existing sample: %s%s"
                             %(name, os.linesep))
            return self[name]
        with h5py._hl.base.phil:
            name, lcpl = self._e(name, lcpl=True)
            gid = h5py.h5g.create(self.id, name, lcpl=lcpl)
            s = Sample(gid, description)
        s.attrs["_is_a_sample"] = True
        #self._samples.append(s.id)
        return s

    def __getitem__(self, name):
        """
            Messing around with h5py class definitions to
            change the dress of the group instances
        """
        o = super(ID01File, self).__getitem__(name)
        if o.attrs.get("_is_a_sample", False):
            return Sample(o.id)
        elif o.attrs.get("_is_a_scan", False):
            return Scan(o.id)
        else:
            return o




def FastEdfCollect(generic_path, idx):
    frame0 = _read_edf_first_acq_nr(generic_path%idx)
    if frame0>0 or not isfile(generic_path%idx):
        raise ValueError("Invalid starting file %s"%(generic_path%idx))
    paths = [generic_path%idx]
    idx += 1
    while True:
        path = generic_path%idx
        if not isfile(path):
            break
        frame0 = _read_edf_first_acq_nr(path)
        if frame0>0:
            paths.append(path)
            idx += 1
        else:
            break
    return paths




