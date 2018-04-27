from __future__ import print_function
from __future__ import division

# global imports
from multiprocessing import Pool
import sys, os
import matplotlib.pyplot as plt
import time
import h5py
import numpy as np
import json
import pyFAI
import time
import nexusformat.nexus as nx
import fabio

##local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice

def integrator(inargs):
               
    fname      = inargs[0]
    maskfname  = inargs[1]
    ponifname  = inargs[2]
    group      = inargs[3]
    no_qbins   = inargs[4]
    timeit     = False

    my_pid = os.getpid()
    print("doing: group %s in process %s" % (group, my_pid))
    f = h5py.File(fname,'r')

    no_frames = f[group].shape[0]
    print('number of frames = %s' % no_frames)
    
    result = np.zeros(shape = (no_frames,2,no_qbins))

    ### setup integrator
    ai = pyFAI.AzimuthalIntegrator()
    ai.load(ponifname)
#    ai.maskfile = maskfname # segmentation fault for pyFAI  0.13.1
    mask = np.copy(fabio.open(maskfname).data)

    ### loop integration:
    if timeit:
        t = time.time()
        
    for frame in range(no_frames):
        print('dataset %s, frame %s of %s' % (group, frame, no_frames))
        data = f[group][frame]
        result[frame,0,:],result[frame,1,:] = ai.integrate1d(data,
                                                             no_qbins,
                                                             mask = mask,
                                                             unit='q_nm^-1',
                                                             safe = False)
                                                             #method = 'lut_ocl')

    if timeit:
        taken_time = time.time()-t
        print('%s frames of dataset %s took %s s on process %s' % (no_frames,
                                                                group,
                                                                taken_time,
                                                                my_pid))
        print('thats %s files per s' % (no_frames/ taken_time))
    
    f.close()

    return result

def integrate1d(args = ['/data/id13/inhouse6/THEDATA_I6_1/d_2016-12-09_user_sc4415_smith/DATA/AUTO-TRANSFER/eiger1/tr5_As40_40p_180um_1315_master.h5',
                        '/data/id13/inhouse6/THEDATA_I6_1/d_2016-12-09_user_sc4415_smith/PROCESS/post_exp/tr5_As40_40p_180um_1315_avg_00000-mask.edf',
                        '/data/id13/inhouse6/THEDATA_I6_1/d_2016-12-09_user_sc4415_smith/PROCESS/SESSION25/OUT/agbeth_calib1_812_avg_00000.poni',
                        '/data/id13/inhouse6/THEDATA_I6_1/d_2016-12-09_user_sc4415_smith/PROCESS/post_exp/tr5_As40_40p_180um_1315_integrated.h5'] ):
    '''
    args[0]: eiger masterfile
    args[1]: maskfile
    args[2]: ponifile
    args[3]: savefile
    '''

    ### input
    
    noprocesses = 13
    no_qbins    = 2000
    timeing     = True

    fname     = args[0]
    maskfname = args[1]
    ponifname = args[2]
    savefname = args[3]

    ### gather info of the datasize
    
    print('opening file %s' % fname)
    f = h5py.File(fname,'r')
    basegroup = 'entry/data/'
    group_list = list(f[basegroup].keys())
    group_list.sort()
    frames_per_file = f[basegroup+group_list[0]].shape[0]
    last_frames = f[basegroup+group_list[-1]].shape[0]
    f.close()

    no_frames = (len(group_list)-1)*frames_per_file + last_frames

    

    ### setting up embarrasingly parrallel process
    todolist = []
    for group_no, group in enumerate(group_list):
        todolist.append([fname,
                         maskfname,
                         ponifname,
                         basegroup + group,
                         no_qbins])

    #### dooing
    if timeing == True:
        t = time.time()
    
    pool = Pool(processes=noprocesses)
    result = pool.map(integrator,todolist)
    result = np.vstack(result)

    
    if timeing == True:
        taken_time = time.time()-t
        print('%s frames took %s s' % (no_frames, taken_time))
        print('thats %s files per s' % (float(no_frames/ taken_time)))
    
    ### initialize the save_file
    if os.path.exists(savefname):
        print('%s already existed, removed')
        os.remove(savefname)
    
    nx_sf = nx.nxload(savefname, 'w')
    nx_entry = nx_sf['entry'] = nx.NXentry()
    nx_data = nx_entry['data'] = nx.NXdata()
    

    ### This is soo not nexus !!! FIX
    nx_data.insert(nx.NXfield(name = 'integrated_data',
                              data = result))

    pool.close()
    nx_sf.flush()
    nx_sf.close()


    return result
