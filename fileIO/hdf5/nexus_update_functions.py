
import sys, os
import h5py
import numpy as np
from nexusformat.nexus import *
import datetime
from silx.io.spech5 import SpecH5

# local import for testing:
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.hdf5.open_h5 import open_h5
import time

  
def update_group_from_file(nx_g,
                           newgroupname,
                           properties = {fname:None}):
    '''
    reads the nx datagroup according to nx_g.name, the idea is to handle .h5 and spec files (ultimately)
    '''
    if newgroupname == 'Eiger4M':
        nx_g = update_from_eiger_master(nx_g,
                                        NXdetector(name=newgroupname),
                                        properties)
        
    elif newgroupname.find('Vortex') > -1:
        nx_g = update_from_xiasettings(nx_g,
                                       NXdetector(name=newgroupname),
                                       properties)

    elif newgroupname == 'spec':
        nx_g = update_initial_spec(nx_g,
                                   NXcollection(name=newgroupname),
                                   properties)

    elif newgroupname == 'calibration':
        nx_g = update_from_ponifile(nx_g,
                                    NXcollection(name=newgroupname),
                                    properties)
        
    elif newgroupname == 'beamline_positioners':
        nx_g = update_motors_from_doolog(nx_g,
                                         NXcollection(name=newgroupname),
                                         properties)
              
    else:
        raise ValueError('unknown group %s' % newgroupname)
    
    return nx_g


def update_from_ponifile(nx_g, new_nx_g, properties = {fname:None}):
    '''
    reads a .poni file into the group
    '''
    poni_fname = properties['fname']
    new_nx_g.insert(nxparse_calibration(poni_fname))
    nx_g.insert(new_nx_g)
    
    return nx_g

def nxparse_calibration(calib_fname):
    nx_calib = NXcollection(name = 'Eiger4M')

    if calib_fname == None:
        calib_fname = '/data/id13/inhouse6/nexustest_aj/files/al2o3_calib1_max.poni'
        print('WARNING: using default dummy file:')

    print('reading calib file')
    print calib_fname + '\n'

    nx_calib.attrs['ponifile_original_path'] = calib_fname
    nx_calib.attrs['ponifile_relative_path'] = os.path.relpath(calib_fname)
    
    calib_f = open(calib_fname, 'r')
    calib_lines = calib_f.readlines()

    for i in range(9):
        current_line = calib_lines[::-1][i]
        name         = current_line.split(':')[0].lstrip().rstrip()
        value        = float(current_line.split(':')[1].lstrip().rstrip())
        units        = default_units(name)
        
        nx_calib.insert(NXfield(name=name,
                                value=value,
                                units=units))

    i = 8
    found = False
    while not found:
        i+=1
        current_line  = calib_lines[::-1][i]
        foundlocation = current_line.find('Calibration done at')
        if foundlocation > 0:
            found = True
            date_str  = current_line[current_line.find('at ')+3::].rstrip()
            datestamp = datetime.datetime.strptime(date_str, '%a %b %d %H:%M:%S %Y').strftime('%Y-%M-%dT%H:%M:%S')
            
            nx_calib.attrs['calibration_time']=datestamp
            
    return nx_calib


def update_calibration_link(nxeiger):
    ''' 
    links between the eiger and the saved calibration
    potentially also implement interpolation
    '''
    print('no calibration information written in the Eiger4M group')
  

def update_initial_spec(nx_g, new_nx_g, properties = {'fname':None}):
    '''
    reads a specfile and finds the next spec run number and the defined counters respective groups in nx_g
    '''
    spec_fname = properties['fname']
    
    if spec_fname == None:
        spec_fname = '/data/id13/inhouse6/nexustest_aj/files/setup.dat'
        print('WARNING: using default dummy file:')
        
    print 'reading file:'
    print spec_fname + '\n' 

    new_nx_g.attrs['specfile_original_path'] = spec_fname
    new_nx_g.attrs['specfile_relative_path'] = os.path.relpath(spec_fname)
    
    sfh5        = SpecH5(spec_fname)
    scan_list   = sfh5.keys()
    scan_no_list = [int(x.split('.')[0]) for x in scan_list]
    scan_no_list.sort()
    last_scan_no =  scan_no_list[-1]
    scan_no      = (last_scan_no + 1)
    next_scan    = '%s.1' % scan_no
    print('WARNING: assuming that the spec scan has not started yet! \nThis scan = last scan +1 = %s' %next_scan)

    new_nx_g.insert(NXfield(name = 'spec_scan_no', value = scan_no))
    new_nx_g.insert(NXfield(name = 'spec_fname', value = spec_fname))
          
    # getting all counters: sfh5['13.1/measurement'].keys()

    nx_g.insert(new_nx_g)
    return nx_g
    
def update_motors_from_doolog(nx_g, new_nx_g, properties = {'fname':None}, verbose=False):
    '''
    specific solution for the ID13 microbranch currently working on a default doolog file
    '''
    doolog_fname = properties['fname']
    
    if doolog_fname == None:
        doolog_fname = '/data/id13/inhouse6/nexustest_aj/files/doolog_001.log'
        print('WARNING: using default dummy file:')
        
    print 'reading file:'
    print doolog_fname + '\n' 

    new_nx_g.attrs['doolog_original_path'] = doolog_fname
    new_nx_g.attrs['doolog_relative_path'] = os.path.relpath(doolog_fname)

    
    log_f = open(doolog_fname, 'r')
    log_lines = log_f.readlines()

    specsession_list = ['bigmic',
                        'scanning',
                        'eybert',
                        'photon',
                        'tuning']

    ### what to update should be set, let's do it!
    
    nx_blm = NXcollection(name = 'positioners_initial')
    
    i = 0
    while len(specsession_list) > 0:

        current_line = log_lines[::-1][i]
        i += 1

        # print 'i = %s' %i       
        if current_line.find('<pos>') > 0:
            for no, session_name in enumerate(specsession_list):
                if current_line.find(session_name) > 0:
                    specsession_list.pop(no)
                    ds = NXcollection(name = session_name)               
                    ds.attrs['log_no']      = int(current_line.split(':')[1].lstrip(' '))
                    ds.attrs['log_time']    = current_line.split(':')[4].lstrip(' ')
                    ds.attrs['update_time'] = "T".join( str( datetime.datetime.now() ).split() )
                    ds.attrs['log_file']    = doolog_fname
                    end = False
                    k = 1
                    while not end:
                        k += 1
                        subline = log_lines[::-1][i-k]
                        if not subline.split(':')[0].lstrip().rstrip() == 'pos':
                            end = True
                        else:
                            value = float(subline.split(':')[-1].split('=')[1].lstrip().rstrip())
                            name  = (subline.split(':')[-1].split('=')[0].lstrip().rstrip())
                            units  = default_units(name)
#                            print 'inserted %s = %s' % (name,value)
                            ds.insert(NXfield(value=value,
                                              name=name,
                                              units=units))

                    nx_blm.insert(ds)

    
    if verbose:
        print('updating group %s from file %s, log entry number %s with:' % (nx_blm.path, doolog_fname, log_no))
        print(list_to_get)

    print('WARNING: motors for sample movements have not been updated')

    new_nx_g.insert(nx_blm)
    nx_g.insert(new_nx_g)
    
    return nx_g


def update_from_xiasettings(nx_g, new_nx_g, properties = {'fname':None},  verbose = False):
    '''
    specific solution for the vortex xia detector, could read these values from a cfg file.
    '''
    xia_fname = properties['fname']
    
    if xia_fname == None:
        xia_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/DATA/AJ2b_after/AJ2b_after/xia_xia00_0480_0000_0005.edf'
        print('WARNING: using default dummy file:')

    print 'reading file:'
    print xia_fname + '\n' 

    xiapath   = os.path.relpath(xia_fname)
    xiaprefix = ('_').join(os.path.basename(xia_fname).split('_')[::-1][3::][::-1])
    
    xiano     = int(os.path.basename(xia_fname).split('_')[::-1][2]) +1

    print('WARNING: assuming that the xia scan has not started yet! \nThis scan = last scan +1 = %s' % xiano)

    new_nx_g.attrs['xia_data_relative_path'] = xiapath
    new_nx_g.attrs['xia_data_original_path'] = xia_fname
    new_nx_g.attrs['xia_data_prefix'] = xiaprefix
    new_nx_g.attrs['xia_data_scannumber'] = xiano
    

    print('writing hard coded default XRF attenuators, including air attenuation!')
    
    nx_attenuators = new_nx_g['attenuators'] = NXcollection()

    deadlayer = NXattenuator(name = 'deadlayer')
    deadlayer.insert(NXfield(name = 'density', value = 2.33, units='g cm-3'))
    deadlayer.insert(NXfield(name = 'material', value = 'Si1'))
    deadlayer.insert(NXfield(name = 'thickness', value = 0.00002, units='cm'))
    deadlayer.insert(NXfield(name = 'Funny_Factor', value = 1))
    
    nx_attenuators.insert(deadlayer)

    atmosphere = NXattenuator(name = 'atmosphere')
    atmosphere.insert(NXfield(name = 'density', value = 0.001205, units='g/cm3'))
    atmosphere.insert(NXfield(name = 'material', value = 'Air'))
    atmosphere.insert(NXfield(name = 'thickness', value = 2.0, units='cm',
                              comment = 'this is not corrected for the position of the detector!!!'))
    atmosphere.insert(NXfield(name = 'Funny_Factor', value = 1))
    
    nx_attenuators.insert(atmosphere)

    window = NXattenuator(name = 'window')
    window.insert(NXfield(name = 'density', value = 1.848, units='g cm-3'))
    window.insert(NXfield(name = 'material', value = 'Be'))
    window.insert(NXfield(name = 'thickness', value = 0.0025, units='cm'))
    window.insert(NXfield(name = 'Funny_Factor', value = 1))

    nx_attenuators.insert(window)
    
    detector = NXattenuator(name = 'detector')
    detector.insert(NXfield(name = 'density', value = 2.33, units='g cm-3'))
    detector.insert(NXfield(name = 'material', value = 'Si'))
    detector.insert(NXfield(name = 'thickness', value = 0.0035, units='cm'))
    detector.insert(NXfield(name = 'Funny_Factor', value = 1))
    
    nx_attenuators.insert(detector)

    nx_attenuators.attrs['log_file']    = None
    nx_attenuators.attrs['update_time'] = "T".join(str(datetime.datetime.now()).split())

    nx_g.insert(new_nx_g)
    
    return nx_g

def update_from_eiger_master(nx_g, new_nx_g, properties = {'fname':None,'list_to_get' : None}, verbose = False):
    '''
    specific solution for eiger master files, get default values from ['entry/instrument/detector'] and data from entry/data/data
    '''

    ### open the eiger master file:

    master_fname = properties['fname']
    if master_fname == None:
        print('WARNING: using default master filename')
        master_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/DATA/AUTO-TRANSFER/eiger1/al2o3_calib1_132_master.h5'

    nx_eiger = nxload(master_fname, 'r')
    nx_eigersource= nx_eiger['entry/instrument/detector']

    ### get the Eigers properties
    
    if list_to_get == None:
        list_to_get = ['bit_depth_image',
                       'bit_depth_readout',
                       'count_time',
                       'countrate_correction_applied',
                       'description',
                       'detector_number',
                       'detector_readout_time',
                       'efficiency_correction_applied',
                       'flatfield_correction_applied',
                       'frame_time',
                       'pixel_mask_applied',
                       'sensor_material',
                       'sensor_thickness',
                       'threshold_energy',
                       'virtual_pixel_correction_applied',
                       'x_pixel_size',
                       'y_pixel_size',
                       'detectorSpecific/data_collection_date']

    if verbose:
        print('updating group %s from file %s with:' % (nx_g.nxpath, master_fname))
        print(list_to_get)
        
    for ds in list_to_get:
        new_nx_g.insert(nx_eigersource[ds])

    
    nx_g.attrs['eiger_master_original_path'] = master_fname
    rel_master_fname = os.path.relpath(master_fname)
    nx_g.attrs['eiger_master_relative_path'] = rel_master_fname
    
    nx_eiger.close()

    ### link the actual data:

#    print 'nx_g.nxroot: %s'%nx_g.nxroot.nxfilename
    
    nx_g.insert(new_nx_g)


    ## this seems dangrous to me, opening the file again:
    f                = h5py.File(nx_g.nxroot.nxfilename)
    local_nxpath     = '/'.join([nx_g.nxpath,new_nx_g.nxname,'data'])
    external_nxpath  = 'entry/data/'

    #    print 'inserting external link at %s\nto file %s\nat the external link %s\n'\
    #        % (local_nxpath, rel_master_fname, external_nxpath)
    
    f[local_nxpath]  = h5py.ExternalLink(rel_master_fname, external_nxpath)

    
    f.close()
  
    
    return nx_g

    
       
def default_units(name):
    angles = ['Theta',
              'Rot1',
              'Rot2',
              'Rot3']
    piezo  = ['nnp1',
              'nnp2',
              'nnp3']
    meter  = ['PixelSize1',
              'PixelSize2',
              'Distance',
              'Poni1',
              'Poni2',
              'Wavelength']
    
    if name in angles:
        units = 'degrees'
    elif name in meter:
        units = 'm'
    elif name in piezo:
        units = 'um'
    else:
        units = 'mm'
    return units
