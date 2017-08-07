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
#       - parallelization
#         (does it make sense if harddisk I/O is the biggest part of the work?)
#       - read/write of 2d slices of .edf cubes faster than whole cube?
#       - define proper nx classes + id01-specific parsing of spec file header
#       - better compatability to xsocs .h5 format
#       - more comments / doc
#
#----------------------------------------------------------------------
"""
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
import itertools
import collections
import numpy as np

import silx.io
from silx.io import specfile
import silx.third_party.EdfFile as EdfFile
import silx.io.fabioh5
from silx.io.spectoh5 import write_spec_to_h5
EdfImage = silx.io.fabioh5.fabio.edfimage.EdfImage



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



_edf_header_types = {
    'ByteOrder': bytes,
    'DataType': bytes,
    'Dim_1': int,
    'Dim_2': int,
    'HeaderID': bytes,
    'Image': int,
    'Offset_1': int,
    'Offset_2': int,
    'Size': int,
    'BSize_1': int,
    'BSize_2': int,
    'ExposureTime': float,
    'Title': bytes,
    'TitleBody': bytes,
    'acq_frame_nb': int,
    'time': np.string_,
    'time_of_day': float,
    'time_of_frame': float
    }



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
                print("setting %s/%s = %s"%(g.name, k, str(v)))
            g[k] = v
    return


def _spech5_deepcopy(scan, origin="/"):
    def helper(s, obj):
        if silx._version.MINOR < 6 and silx._version.MAJOR == 0:
            s = os.path.relpath(s, origin)
        if not s:
            return
        if isinstance(obj, silx.io.spech5.SpecH5Dataset):
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
    lines = header.splitlines()
    acq_frame_nb = [s for s in lines if s.startswith(b'acq_frame_nb')][0]
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
    
    
    def addEdfFile(self, edf, detectornum=None, compr_lvl=6):
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
            edf = EdfFile.EdfFile(edf, fastedf=True)
        
        if not edf.NumImages:
            return
        
        newshape = (edf.NumImages, edf.Images[0].Dim1, edf.Images[0].Dim2)
        if self.debug:
            print("Loaded image. Shape %ix%ix%x"%newshape)
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
                if not key in self["instrument"]:
                    break
                if self["instrument"][key]["data"].shape[1:] == newshape[1:]:
                    break
            detectornum = i
        else:
            key = "detector_%i"%detectornum
        
        if not key in self["instrument"]:
            detector = self["instrument"].create_group(key)
            detector.create_dataset("data", 
                                  newshape,
                                  maxshape=(None,newshape[1],newshape[2]),
                                  compression="gzip",
                                  compression_opts=compr_lvl,
                                  dtype=data.dtype) # Todo: think about chunks
            detector["data"][...] = data
            
            # rocess metadata
            others = detector.create_group("others")
            #_makedefaults(others, det["others"]) # creates non-resizable stuff
            for name, value in d.items(): # copy all metadata
                if self.debug:
                    print(name)
                if name in _edf_header_types:
                    mdtype = _edf_header_types[name]
                    if mdtype is bytes:
                        mdtype = h5py.special_dtype(vlen=bytes)
                    else:
                        value = np.array(value, dtype=mdtype)
                        if mdtype is np.string_:
                            mdtype = value.dtype
                else:
                    mdtype = bytes

                others.create_dataset(name,
                                      (len(value),),
                                      maxshape=(None,),
                                      dtype=mdtype)
                others[name][...] = value
            # silx compatability:
            #self.copy("instrument/%s"%key, "measurement/image_%i"%detectornum)
            # hard link:
            self["measurement/image_%i"%detectornum] = self["instrument/%s"%key]
            
        else:
            # do the resizing
            detector = self["instrument"][key]
            newlen = detector["data"].shape[0] + newshape[0]
            detector["data"].resize((newlen,newshape[1],newshape[2]))
            detector["data"][-newshape[0]:] = data
            if self.debug:
                print("New dataset size: %ix%ix%i"%detector["data"].shape)
            # process metadata
            others = detector["others"]
            for name, value in d.items(): # copy all metadata
                if name in _edf_header_types:
                    mdtype = _edf_header_types[name]
                    #if not mdtype in (bytes, np.string_):
                    value = np.array(value, dtype=mdtype)
                newlen = others[name].shape[0] + len(value)
                others[name].resize((newlen,)) # strictly 1d
                others[name][-len(value):] = value



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
    
    def fetch_edf_spec(self, pathonly=False, verbose=True, **edf_kw):
        fast = self.attrs["_fast"]
        header = self["instrument/specfile/scan_header"].value.splitlines()
        if fast:
            #impath = [s for s in header if b"imageFile" in s][0]
            impath = [s for s in header if s.startswith(b"#C imageFile")]
            if not impath:
                return []
            impath = impath[0]
            impath = impath.split()[2:]
            impath = dict((s.strip(b"]").split(b"[") for s in impath))
            generic_path = os.path.join(
                    impath[b"dir"],
                    impath[b"prefix"] + impath[b"idxFmt"] + impath[b"suffix"]
                            )
            idx = int(impath[b"nextNr"])
            #impath = generic_path%idx
            detname = ""
            all_paths = FastEdfCollect(generic_path.decode(), idx)
        else:
            impath = [s for s in header if s.startswith(b"#ULIMA_")][0]
            detname, impath = impath.split()
            detname = detname[7:]
            inr = self["measurement/mpx4inr"]
            if not inr.len():
                return []
            startnr = int(inr[0])
            impath = impath.decode().replace("_%05d"%startnr, "_%05d")
            all_paths = [impath%i for i in inr]
        
        all_paths = np.array(all_paths, dtype=np.string_)
        #dt = h5py.special_dtype(vlen=bytes)
        self["measurement"].create_dataset("image_files",
                                           shape=all_paths.shape,
                                           dtype=all_paths.dtype,
                                           data=all_paths)

        for impath in all_paths:
            impath = impath.decode()
            if not isfile(impath):
                print("Warning: File not found: %s. Skipping..."%impath)
                continue
            if pathonly:
                if not verbose:
                    break
                print("  Found path %s"%impath)
            else:
                try:
                    if verbose:
                        print("  Fetching %s"%impath)
                    self.addEdfFile(impath, **edf_kw)
                except Exception as emsg:
                    print("Could not load file: %s"%emsg)
        return all_paths





class Sample(h5py.Group):
    scans = []
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
            name = '%i.0'%number
        else:
            name = number
            number = int(name.split(".")[0])
        
        if name not in self:
            with h5py._hl.base.phil:
                # new scan
                name, lcpl = self._e(name, lcpl=True)
                gid = h5py.h5g.create(self.id, name, lcpl=lcpl)
                scan = Scan(gid)
                scan.attrs["_is_a_scan"] = True
                if name.decode().split(".")[1]=="0":
                    self.scans.append(number)
        else:
            # select scan
            scan = Scan(self[name].id)
        
        for f in datafiles:
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
    
    
    
    def addSpecScan(self, specscan, number=None, 
                          fetch_edf=True, verbose=True, **edf_kw):
        if not isinstance(specscan, silx.io.spech5.SpecH5Group):
            raise ValueError("Need `silx.io.spech5.SpecH5Group` as spec scan input.")
        filename = specscan.file.filename
        #fast = filename.split("_")[-2] == "fast"
        fast = "_fast_" in os.path.basename(filename)
        root = specscan.name.split("/")[1]
        specnumber = int(root.split(".")[0]) #+ (1 if fast else 0)
        
        if number is None:
            if fast:
                number = int(os.path.splitext(filename)[0][-5:])
            else:
                number = specnumber
        
        
        if isinstance(number, int):
            if fast:
                name = '%i.0.kmap_%05i'%(number, specnumber)
            else:
                name = '%i.1'%number
        else:
            name = number
        
        #print(specnumber)
        scan = self.addScan(name)
        
        if "title" in scan:
            print("Warning: omitting %s (already exists)."%(name))
            return
        
        scan.attrs["_fast"] = fast
        
        if verbose:
            print("Importing spec scan %s from %s to %s..."%(root, filename, scan.name))
        
        #specscan.copy(specname, self, name=name) #copying does not work with spech5
        # will be replaced by spectoh5.SpecToHdf5Writer:
        specscan.visititems(_spech5_deepcopy(scan, specscan.name))
        
        if fetch_edf:
            scan.fetch_edf_spec(verbose=verbose, **edf_kw)
        
        return scan
    
    
    
    def importSpecFile(self, specfile, numbers=(), newnumbers=(), **addSpec_kw):
        if isfile(specfile):
            s5f = silx.io.open(specfile)
        elif isinstance(specfile, silx.io.spech5.SpecH5):
            s5f = specfile
        else:
            if isinstance(specfile, str):
                raise TypeError("File not found: %s"%specfile)
            else:
                raise TypeError("Input type not supported: %s"%str(type(specfile)))
        
        if not numbers:
            numbers = range(1, len(s5f)+1)
        if isinstance(numbers[0], int):
            gen = (s5f[i-1] for i in numbers)
        else:
            gen = (s5f[i] for i in numbers)
        
        
        for i, scan in enumerate(gen):
            self.addSpecScan(scan,
                             newnumbers[i] if newnumbers else None,
                             **addSpec_kw)
    
    
    
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




if __name__=="__main__":
    ### TEST:
    
    datadir = "/data/visitor/ma3331/id01/"
    fastspecfile = "knno-47-008-GSO_fast_%05i.spec"%13
    fastpath = os.path.join(datadir, fastspecfile)
    specpath = "/data/visitor/ma3331/id01/knno-47-008-GSO.spec"
    
    
    if not isfile(fastpath):
        raise IOError("File not found: %s"%fastpath)
    
    sf = specfile.SpecFile(fastpath)
    for scan in sf:
        impath = [s for s in scan.scan_header if "imageFile" in s][0]
        impath = impath.split()[2:]
        impath = dict((s.strip("]").split("[") for s in impath))
        generic_path = os.path.join(impath["dir"],
                                    impath["prefix"] + impath["idxFmt"] + impath["suffix"])
        idx = int(impath["nextNr"])
        print(generic_path, idx)
    
    
    
    
    test = ID01File("/tmp/test2.h5")
    test.clear()
    a = test.addSample("GS2", "irgendwas")
    a = test["GS2"]
    b = a.addScan(1)#, e,e,e,e)
    b = a["1.0"]
    c = test.addSample("GS3", "irgendwas")
    print(a["1.0/instrument"])
    
    b.debug = True
    if 0:
        paths = FastEdfCollect(generic_path, 11)
        edf = EdfFile(paths[0]) #pick one
        b2 = a.addScan(1, edf) #this is another way to do it
        b.addEdfFile(edf.FileName) # add something big
        dat = test["GS2/1.0/instrument/detector_0"]["data"]
        print(b==b2) # should be true
        print(b["instrument/detector_0/others/time"].value) # some metadata
    
    #a.importSpecFile(specpath)
    
    a.importSpecFile(fastpath)
    #sm = e["scan_0/instrument/detector_0"]["others"]
    #print sm.keys() # all columns of metadata
    
    test.close()
    




