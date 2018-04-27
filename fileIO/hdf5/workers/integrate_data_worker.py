from __future__ import print_function

import numpy as np
import h5py
import os, sys
import pyFAI

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy

def integrate_data_employer(pickledargs_fname):
    '''
    gutwrenching way to (almost) completely uncouple the h5 access from the motherprocess
    Can be used to multiprocess.pool control the workers.
    '''
    fname =  __file__
    cmd = 'python {} {}'.format(fname, pickledargs_fname)

    # import inspect
    # linno = inspect.currentframe().f_lineno
    # print('DEBUG:\nin ' + __file__ + '\nline '+str(linno))
    # print(cmd)

    
    os_response = os.system(cmd)
    if os_response >1:
        raise ValueError('in {}\nos.system() has responded with errorcode {} in process {}'.format(fname, os_response, os.getpid))
    

def integrate_data_worker(pickledargs_fname):
    '''
    interates troi into target_fname[target_group][XXX/data][target_index] from source_name[source_datasetpath][source_index][troi]
    with XXX = 'q_integration_1D','q_integration_2D' and 'tth_integration_2D'. all 3!
    these dataset have to allready exist with the right shape and dtype and no compression!
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)
    # import inspect

    # linno = inspect.currentframe().f_lineno
    # print('DEBUG:\nin ' + __file__ + '\nline '+str(linno))
    # print(unpickled_args)

    target_fname = unpickled_args[0]
    target_grouppath = unpickled_args[1]
    target_index_list = unpickled_args[2]
    source_fname = unpickled_args[3]
    source_datasetpath = unpickled_args[4]
    source_index_list = unpickled_args[5]
    troi = unpickled_args[6]
    troiponi_fname = unpickled_args[7]
    npt_azim, npt_rad = unpickled_args[8]
    
    verbose = unpickled_args[9]

    if len(unpickled_args)>10:
        timer_fname = unpickled_args[10]
    else:
        timer_fname = None

        
    if verbose:
        print('='*25)
        print('process {} is doing'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)
    
    if target_fname == source_fname:
        with h5py.File(target_fname) as h5_file:

            target_group = h5_file[target_grouppath]
            frameshape = h5_file[source_datasetpath][source_index_list[0]].shape

            if type(troi) == type(None):
                slices = troi_to_slice(((0,0),(frameshape[0],frameshape[1])))
            else:
                slices = troi_to_slice(troi)
            
            ai = pyFAI.AzimuthalIntegrator()
            ai.reset()
            # might give error for multi-access that could be caught:
            ai.load(troiponi_fname)
            # BUG see pyFAI mail 28.03, fixes in pyFAI.version > 0.16
            ai.setChiDiscAtZero()
            
            if 'mask' in target_group.keys():
                mask = target_group['mask'].value
            else:
                mask = np.zeros(shape = (frameshape[0],frameshape[1]))

            for target_index, source_index in zip(target_index_list,source_index_list):
                
                raw_data = h5_file[source_datasetpath][source_index][slices[0],slices[1]]

                q2d_data, dummy1, dummy2 = ai.integrate2d(data = raw_data, mask = mask, npt_azim = npt_azim ,npt_rad = npt_rad , unit='q_nm^-1', polarization_factor=0.99)

                target_group['q_2D/data'][target_index] = q2d_data
                I_azim = q2d_data.sum(1)
                target_group['chi_azimuthal/I'][target_index,:] = I_azim
                I_radial = q2d_data.sum(0)
                target_group['q_radial/I'][target_index,:] = I_radial

                tth2d_data, dummy1, dummy2 = ai.integrate2d(data = raw_data, mask = mask, npt_azim = npt_azim ,npt_rad = npt_rad , unit='2th_deg', polarization_factor=0.99)

                target_group['tth_2D/data'][target_index] = tth2d_data
                I_tth = tth2d_data.sum(0)
                target_group['tth_radial/I'][target_index,:] = I_tth
                
                if verbose:
                    print('process {} is integrating frame {}'.format(os.getpid(),target_index))

            # h5_file.flush()
            
    else:
        # not tested
        with h5py.File(target_fname) as target_file:
            with h5py.File(source_fname) as source_file:
                
                target_group = target_file[target_grouppath]
                frameshape = source_file[source_datasetpath][source_index_list[0]].shape

                if 'mask' in target_group.keys():
                    mask = target_group['mask'].value
                else:
                    mask = np.zeros(shape = (frameshape[0],frameshape[1]))

                for target_index, source_index in zip(target_index_list,source_index_list):

                    raw_data = source_file[source_datasetpath][source_index][slices[0],slices[1]]

                    q2d_data, dummy1, dummy2 = ai.integrate2d(data = raw_data, mask = mask, npt_azim = npt_azim ,npt_rad = npt_rad , unit='q_nm^-1')

                    target_group['q_2D/data'][target_index] = q2d_data
                    I_azim = q2d_data.sum(1)
                    target_group['chi_azimuthal/I'][target_index,:] = I_azim
                    I_radial = q2d_data.sum(0)
                    target_group['q_radial/I'][target_index,:] = I_radial
                
                    tth2d_data, dummy1, dummy2 = ai.integrate2d(data = raw_data, mask = mask, npt_azim = npt_azim ,npt_rad = npt_rad , unit='2th_deg')

                    target_group['tth_2D/data'][target_index] = tth2d_data
                    I_tth = tth2d_data.sum(0)
                    target_group['tth_radial/I'][target_index,:] = I_tth
                    
            
                    if verbose:
                        print('process {} is integrating frame {}'.format(os.getpid(),target_index))


            # target_file.flush()
            
    if verbose:
        print('process {} is done'.format(os.getpid()))
        print('='*25)

    
    if timer_fname != None:
        os.remove(timer_fname)


if __name__=='__main__':
    ''' 
    This is used by the local function integrate_troi_employer(pickledargs_fname),
    DO NOT CHANGE
    '''
    if len(sys.argv)!=2:
        print('usage : python integrate_troi_worker <pickled_instruction_list_fname>')
    pickledargs_fname = sys.argv[1]
    integrate_data_worker(pickledargs_fname)
                        
