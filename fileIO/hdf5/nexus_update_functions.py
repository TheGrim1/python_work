from __future__ import print_function
from __future__ import division

import sys, os
import h5py
import numpy as np
import nexusformat.nexus as nx
import datetime
from silx.io.spech5 import SpecH5

# local import for testing:
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.hdf5.open_h5 import open_h5
from fileIO.hdf5.nexus_tools import id13_default_units as default_units
import time

  
def update_group_from_file(nxentry,
                           newgroupname,
                           properties = {'fname':None}):
    '''
    passes nx_entry, newgroupname and the properties dict to the right update function
    '''
    if newgroupname == 'Eiger4M':
        nx_g = nxentry['instrument']
        nx_g = update_from_eiger_master(nx_g,
                                        properties)
        
    elif newgroupname.find('xia') > -1:
        nx_g = nxentry['instrument']
        nx_g = update_from_xiasettings(nx_g,
                                       properties)

    elif newgroupname == 'spec':
        nx_g = nxentry['instrument']
        nx_g = update_initial_spec(nx_g,
                                   properties)

    elif newgroupname == 'calibration':
        nx_g = nxentry['instrument']
        nx_g = update_from_ponifile(nx_g,
                                    properties)
        
    elif newgroupname == 'beamline_positioners':
        nx_g = nxentry['instrument']
        nx_g = update_motors_from_doolog(nx_g,
                                         properties)

    elif newgroupname == 'initial_command':
        nx_g = nxentry['instrument']
        nx_g = update_command(nx_g,
                              properties)
    elif newgroupname == 'sample_info':
        nx_g = nxentry['sample']
        nx_g = update_sample_info(nx_g,
                                  properties)
        
    else:
        raise ValueError('unknown group %s' % newgroupname)
    
    return nx_g


def update_sample_info(nx_g, properties = {'auto_update' : True}):
    '''
    writes some default information if auto_update: 
    TODO: eiger_prefix, eiger, scan_no, spec_scanno

    and updates the given sample info in entry/sample/sample_info
    '''
    for attr, value in list(properties.items()):
        nx_g.attrs[attr] = value

    if properties['auto_update'] :
        print('auto update sample information is not yet implemented')

    
    return nx_g

def update_command(nx_g, properties = {'cmd'    :'dscan dummy 0 1 1',
                                                 'type'   : 'eiger',
                                                 'version':'Eiger_3_Apples.py',
                                                 'motors' :[['dummy',0.0,1.0,1]]}):
    '''
    stores command meta data and expected motor scan motor positions
    '''
    new_nx_g = nx.NXcollection(name = 'initial_command')
    new_nx_g.attrs['command']  = properties['cmd']
    new_nx_g.attrs['type']     = properties['type']
    new_nx_g.attrs['version']  = properties['version']

    new_nx_g.insert(nx.NXfield(value = properties['exp_time'],
                               name  = 'exp_time',
                               units = 's'))
   

    if 'motors' in list(properties.keys()):
        parsed_motors = properties['motors']
    else:
        ## TODO parse the command 
        raise NotImplementedError('parsing motors from command strings is currently not implemented')

    axes_list = []
    for parsed_motor in parsed_motors:
        motor = parsed_motor[0]
        start = float(parsed_motor[1])
        stop  = float(parsed_motor[2])
        steps = int(parsed_motor[3])

        
        positions = np.arange(start,stop+((stop-start)/steps),((stop-start)/steps))
        new_nx_g.insert(nx.NXpositioner(name = motor,
                                        value= positions,
                                        unit = default_units(motor)))
        axes_list.append(motor)
    axesstr = (':').join(axes_list)
    new_nx_g.attrs['axes'] = axesstr
    
    nx_g.insert(new_nx_g)
    return nx_g

def update_from_ponifile(nx_g, properties = {'fname':None}, groupname=None):
    '''
    reads a .poni file into the group
    '''
    new_nx_g = nx.NXcollection(name='calibration')
    poni_fname = properties['fname']
    groupname  = groupname   
    new_nx_g.insert(nxparse_calibration(poni_fname, name=groupname))
    nx_g.insert(new_nx_g)
    
    return nx_g

def nxparse_calibration(calib_fname, name = None, verbose = False):
    if name == None:
        nx_calib = nx.NXcollection(name = 'Eiger4M')
    else:
        nx_calib = nx.NXcollection(name = name)

    if calib_fname == None:
        calib_fname = '/data/id13/inhouse6/COMMON_DEVELOP/py_andreas/nexustest_aj/files/al2o3_calib1_max.poni'
        print('WARNING: using default dummy file:')
        
    if verbose:
        print('reading calib file')
        print(calib_fname + '\n')

    nx_calib.attrs['ponifile_original_path'] = calib_fname
    nx_calib.attrs['ponifile_relative_path'] = os.path.relpath(calib_fname)
    
    calib_f = open(calib_fname, 'r')
    calib_lines = calib_f.readlines()
    
    for current_line in calib_lines[::-1][0:9]:

        name         = current_line.split(':')[0].lstrip().rstrip()
        try:
            value        = float(current_line.split(':')[1].lstrip().rstrip())
            units        = default_units(name)
        
            nx_calib.insert(nx.NXfield(name=name,
                                       value=value,
                                       units=units))
        except ValueError:
            pass

    i = 8
    found = False
    while not found:
        i+=1
        current_line  = calib_lines[::-1][i]
        foundlocation = current_line.find(' at ')
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
  

def update_initial_spec(nx_g, properties = {'fname':None}):
    '''
    reads a specfile and finds the next spec run number and the defined counters respective groups in nx_g
    '''
    new_nx_g = nx.NXcollection(name='spec')
    spec_fname = properties['fname']
    
    if spec_fname == None:
        spec_fname = '/data/id13/inhouse6/COMMON_DEVELOP/py_andreas/nexustest_aj/files/setup.dat'
        print('WARNING: using default dummy file:')
        
    print('reading file:')
    print(spec_fname + '\n') 

    new_nx_g.attrs['specfile_original_path'] = spec_fname
    new_nx_g.attrs['specfile_relative_path'] = os.path.relpath(spec_fname)

    if not properties['spec_scan_no_next'] == None:
        scan_no =  properties['spec_scan_no_next']
    else:
        sfh5        = SpecH5(spec_fname)
        scan_list   = sfh5.keys()
        scan_no_list = [int(x.split('.')[0]) for x in scan_list]
        scan_no_list.sort()
        last_scan_no =  scan_no_list[-1]
        scan_no      = (last_scan_no + 1)
        next_scan    = '%s.1' % scan_no
        print('WARNING: assuming that the spec scan has not started yet! \nThis scan = last scan +1 = %s' %next_scan)

    new_nx_g.insert(nx.NXfield(name = 'spec_scan_no_next', value = scan_no))
    new_nx_g.insert(nx.NXfield(name = 'spec_fname', value = spec_fname))
          
    # getting all counters: sfh5['13.1/measurement'].keys()

    nx_g.insert(new_nx_g)
    
    return nx_g

def update_final_spec(nx_g, properties = {'fname':None, 'scanno':None}, verbose = False):
    '''
    reads a specfile and finds the scanno
    adds defined counters into respective groups in nx_g
    '''
    
    new_nx_g = nx.NXcollection(name='spec')
    spec_fname = properties['fname']
    scanno = properties['scanno']
    scan_group = '{}.1'.format(scanno)

    
    if spec_fname == None:
        spec_fname = '/data/id13/inhouse6/COMMON_DEVELOP/py_andreas/nexustest_aj/files/setup.dat'
        print('WARNING: using default dummy file:')

    if verbose:
        print('reading file:')
        print(spec_fname)
        print('scan number {}\n'.format(scanno))

    specfile_open = False
    while not specfile_open:
        try:
            sfh5 = SpecH5(spec_fname)
            specfile_open = True
        except ValueError as msg:
            print('*'*25)
            print(msg)
            print('Got ValueError opening specfile, retrying')
            print('*'*25)
            
    # sfh5 = SpecH5(spec_fname)
    scan_list   = list(sfh5.keys())

    if scan_group not in scan_list:
        print(scan_list)
        raise ValueError('Scan number {} not found in scan list printed above'.format(scan_group))

    new_nx_g.attrs['specfile_original_path'] = spec_fname
    new_nx_g.attrs['specfile_relative_path'] = os.path.relpath(spec_fname)

    new_nx_g.insert(nx.NXfield(name = 'spec_scanno', value = scanno))

    # sloshing over all counters and positioners

    meas_group = scan_group+'/measurement'
    nx_measurement = nx.NXcollection(name='measurement')
    counter_list = sfh5[meas_group].keys()    
    for counter in counter_list:
        counter_group = meas_group+'/'+counter 
        nx_measurement.insert(nx.NXfield(name = counter, value = sfh5[counter_group].value))

    nx_g.insert(nx_measurement)    

    if verbose:
        print('updating with counters:')
        print(counter_list)

    pos_group = scan_group+'/instrument/positioners'
    nx_positioners = nx.NXcollection(name='positioners')
    positioner_list = sfh5[pos_group].keys()  
    for positioner in positioner_list:
        positioner_group = pos_group+'/'+positioner 
        nx_positioners.insert(nx.NXfield(name = positioner, value = sfh5[positioner_group].value))

    if verbose:
        print('updating with positioners:')
        print(positioner_list)

    nx_g.insert(nx_positioners)

    new_nx_g.insert(nx.NXfield(name='start_time', value = sfh5[scan_group]['start_time']))
    new_nx_g.insert(nx.NXfield(name='title', value = sfh5[scan_group]['title']))

    nx_g.insert(new_nx_g)

    sfh5.close()

        
    return nx_g

def update_motors_from_doolog(nx_g, properties = {'fname':None}, verbose=False):
    '''
    specific solution for the ID13 microbranch currently working on a default doolog file
    '''
    new_nx_g = nx.NXcollection(name='beamline_positioners')
    doolog_fname = properties['fname']
    
    if doolog_fname == None:
        doolog_fname = '/data/id13/inhouse6/COMMON_DEVELOP/py_andreas/nexustest_aj/files/doolog_001.log'
        print('WARNING: using default dummy file:')
        
    print('reading file:')
    print(doolog_fname + '\n') 

    new_nx_g.attrs['doolog_original_path'] = doolog_fname
    new_nx_g.attrs['doolog_relative_path'] = os.path.relpath(doolog_fname)

    
    log_f = open(doolog_fname, 'r')
    log_lines = log_f.readlines()

    if properties['instrument_name'] == 'id13_eh2':
        specsession_list = ['bigmic',
                            'scanning',
                            'eybert',
                            'photon',
                            'tuning']
    elif properties['instrument_name'] == 'id13_eh3':
        specsession_list = ['EH3',
                            'eybert',
                            'photon',
                            'tuning']
    else:
        raise ValueError('%s is not a valid instrument name' %properties['instrument_name'])
                
        
    ### what to update should be set, let's do it!
    
    nx_blm = nx.NXcollection(name = 'positioners_initial')
    
    i = 0
    while len(specsession_list) > 0:

        current_line = log_lines[::-1][i]
        i += 1

        # print 'i = %s' %i       
        if current_line.find('<pos>') > 0:
            for no, session_name in enumerate(specsession_list):
                if current_line.find(session_name) > 0:
                    specsession_list.pop(no)
                    ds = nx.NXcollection(name = session_name)               
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
                            ds.insert(nx.NXpositioner(value=value,
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


def update_from_xiasettings(nx_g, properties = {'fname':None},  verbose = False):
    '''
    specific solution for the vortex xia detectors, could read these values from a cfg file.
    '''
    new_nx_g = nx.NXdetector(name='xia')
    xia_fname = properties['fname']
    
    if xia_fname == None:
        xia_fname = '/data/id13/inhouse6/COMMON_DEVELOP/py_andreas/nexustest_aj/files/xia_xia00_0480_0000_0005.edf'
        print('WARNING: using default dummy file:')

    print('reading file:')
    print(xia_fname + '\n') 

    xiapath   = os.path.relpath(xia_fname)
    xiaprefix = ('_').join(os.path.basename(xia_fname).split('_')[::-1][3::][::-1])
    
    xiano     = int(os.path.basename(xia_fname).split('_')[::-1][2]) +1

    print('WARNING: assuming that the xia scan has not started yet! \nThis scan = last scan +1 = %s' % xiano)

    new_nx_g.attrs['xia_data_relative_path'] = xiapath
    new_nx_g.attrs['xia_data_original_path'] = xia_fname
    new_nx_g.attrs['xia_data_prefix'] = xiaprefix
    new_nx_g.attrs['xia_data_next_scannumber'] = xiano
    
    print('writing hard coded default XRF attenuators, including air attenuation!')
    
    nx_attenuators = new_nx_g['attenuators'] = nx.NXcollection()

    deadlayer = nx.NXattenuator(name = 'deadlayer')
    deadlayer.insert(nx.NXfield(name = 'density', value = 2.33, units='g cm-3'))
    deadlayer.insert(nx.NXfield(name = 'material', value = 'Si1'))
    deadlayer.insert(nx.NXfield(name = 'thickness', value = 0.00002, units='cm'))
    deadlayer.insert(nx.NXfield(name = 'Funny_Factor', value = 1))
    
    nx_attenuators.insert(deadlayer)

    atmosphere = nx.NXattenuator(name = 'atmosphere')
    atmosphere.insert(nx.NXfield(name = 'density', value = 0.001205, units='g/cm3'))
    atmosphere.insert(nx.NXfield(name = 'material', value = 'Air'))
    atmosphere.insert(nx.NXfield(name = 'thickness', value = 2.0, units='cm',
                              comment = 'this is not corrected for the position of the detector!!!'))
    atmosphere.insert(nx.NXfield(name = 'Funny_Factor', value = 1))
    
    nx_attenuators.insert(atmosphere)

    window = nx.NXattenuator(name = 'window')
    window.insert(nx.NXfield(name = 'density', value = 1.848, units='g cm-3'))
    window.insert(nx.NXfield(name = 'material', value = 'Be'))
    window.insert(nx.NXfield(name = 'thickness', value = 0.0025, units='cm'))
    window.insert(nx.NXfield(name = 'Funny_Factor', value = 1))

    nx_attenuators.insert(window)
    
    detector = nx.NXattenuator(name = 'detector')
    detector.insert(nx.NXfield(name = 'density', value = 2.33, units='g cm-3'))
    detector.insert(nx.NXfield(name = 'material', value = 'Si'))
    detector.insert(nx.NXfield(name = 'thickness', value = 0.0035, units='cm'))
    detector.insert(nx.NXfield(name = 'Funny_Factor', value = 1))
    
    nx_attenuators.insert(detector)

    nx_attenuators.attrs['log_file']    = None
    nx_attenuators.attrs['update_time'] = "T".join(str(datetime.datetime.now()).split())

    nx_g.insert(new_nx_g)
    
    return nx_g

def update_from_eiger_master(nx_g, properties = {'fname':None}, verbose = False):
    '''
    specific solution for eiger master files, get default values from ['entry/instrument/detector'] and data from entry/data/data
    something sometimes goes wrong with the h5 file containing nx_g after adding the hard links.
    RuntimeError: Unable to create link (can't locate ID)
    No idea why, but closing and reopening the file fixes the issue.
    '''

    ### open the eiger master file:
    
    new_nx_g = nx.NXdetector(name='Eiger4M')
    
    master_fname = properties['fname']
    if master_fname == None:
        print('WARNING: using default master filename')
        master_fname = '/data/id13/inhouse6/COMMON_DEVELOP/py_andreas/nexustest_aj/files/setup1_8_master.h5'

    if os.path.exists(master_fname):
        if verbose:
            print('found master file at ' + master_fname)
    else:
        raise ValueError('master file not found at ' + master_fname)
        
    nx_eiger = nx.nxload(master_fname, 'r')
    nx_eigersource= nx_eiger['entry/instrument/detector']

    ### get the Eigers properties
    
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

    
    new_nx_g.attrs['eiger_master_original_path'] = master_fname
    rel_master_fname = os.path.relpath(master_fname)
    new_nx_g.attrs['eiger_master_relative_path'] = rel_master_fname
    
    nx_eiger.close()

    ### link the actual data:

#    print 'nx_g.nxroot: %s'%nx_g.nxroot.nxfilename
    
    nx_g.insert(new_nx_g)

    ### nexusformat external links:
    local_nxpath     = new_nx_g.nxname + '/data'
    external_nxpath  = '/entry/data/'
    # print local_nxpath
    nx_g[local_nxpath] = nx.NXlink(external_nxpath, file = master_fname)
    nx_g[local_nxpath].attrs['sigal'] = 'data'
    
    # ## this seems dangrous to me, opening the file again:
    # f                = h5py.File(nx_g.nxroot.nxfilename)
    # local_nxpath     = '/'.join([nx_g.nxpath,new_nx_g.nxname,'data'])
    # external_nxpath  = 'entry/data/'
    # f[local_nxpath]  = h5py.ExternalLink(rel_master_fname, external_nxpath)    
    # f.close()
  
    
    return nx_g

def update_from_id01_master(nx_g, properties = {'fname':None}, verbose = False):
    '''
    specific solution for master files created from id01 edf files with edf_to_master, get default values from ['entry/instrument/detector'] and data from entry/data/data
    something sometimes goes wrong with the h5 file containing nx_g after adding the hard links.
    '''

    ### open the eiger master file:
    
    new_nx_g = nx.NXdetector(name='maxipix')
    master_fname = properties['fname']
    if master_fname == None:
        print('WARNING: using default master filename')
        master_fname = '/data/id13/inhouse6/COMMON_DEVELOP/py_andreas/nexustest_aj/files/setup1_8_master.h5'

    if os.path.exists(master_fname):
        if verbose:
            print('found master file at ' + master_fname)
    else:
        raise ValueError('master file not found at ' + master_fname)
        
    nx_eiger = nx.nxload(master_fname, 'r')
    nx_eigersource= nx_eiger['entry/instrument/detector']

    ### get the Eigers properties
    
    list_to_get = ['description',
                   'x_pixel_size',
                   'y_pixel_size']

    if verbose:
        print('updating group %s from file %s with:' % (nx_g.nxpath, master_fname))
        print(list_to_get)
        
    for ds in list_to_get:
        new_nx_g.insert(nx_eigersource[ds])

    
    new_nx_g.attrs['maxipix_master_original_path'] = master_fname
    rel_master_fname = os.path.relpath(master_fname)
    new_nx_g.attrs['maxipix_master_relative_path'] = rel_master_fname
    
    nx_eiger.close()

    ### link the actual data:

#    print 'nx_g.nxroot: %s'%nx_g.nxroot.nxfilename
    
    nx_g.insert(new_nx_g)

    ### nexusformat external links:
    local_nxpath     = new_nx_g.nxname + '/data'
    external_nxpath  = '/entry/data/'
    # print local_nxpath
    nx_g[local_nxpath] = nx.NXlink(external_nxpath, file = master_fname)
    nx_g[local_nxpath].attrs['sigal'] = 'data'
    
    return nx_g



def insert_dataset(nx_g,
                   data = np.random.randint(5,10,7),
                   **kwargs):
    if 'name' in kwargs:
        name = kwargs['name']
    else:
        name = 'data'

    if 'units' in kwargs:
        units = kwargs['units']
    else:
        units = default_units(name)

    print('data')
    print(data)
    print('kwargs')
    print(kwargs)
    nx_g.insert(nx.NXfield(value = data, units = units, name = name))
    
    if 'axes' in kwargs:
        nx_g.attrs['signal'] = name
        axes = []
        for axis, properties in kwargs['axes']:
            print('appending axis %s' % axis)
            print('with properties:')
            print(properties)
            axes.append(axis)
            if 'link' in properties:
                nx_g.makelink(nx_g.nxroot[properties['link']])
            elif 'values' in properties:
                values = properties['values']

                if 'units' in properties:
                    units = properties['units']
                else:
                    units = default_units(axis)
                nx_g[axis] = nx.NXfield(value = values, units = units)
            else:
                nx_g.makelink(nx_g.nx_root['entry/measurement/' + axis])
                    
        nx_g.attrs['axes'] = ':'.join(axes)
    
    return nx_g


        
