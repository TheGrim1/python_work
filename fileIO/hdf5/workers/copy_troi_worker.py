from __future__ import print_function

import numpy as np
import h5py
import os, sys

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy

def copy_troi_employer(pickledargs_fname):
    '''
    gutwrenching way to completely uncouple the h5 access from the motherprocess
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

    

def copy_troi_worker(pickledargs_fname):
    '''
    copies troi into target_fname[target_datasetpath][target_index] from source_name[source_datasetpath][source_index][troi]
    these dataset have to allready exist with the right shape and dtype 
    no compression, if more than on onf these workers is working on one file! 
    Changes to unpickling here must be updated in h5_scan_nexusversion
    '''
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)
    # import inspect

    # linno = inspect.currentframe().f_lineno
    # print('DEBUG:\nin ' + __file__ + '\nline '+str(linno))
    # print(unpickled_args)

    target_fname = unpickled_args[0]
    target_datasetpath = unpickled_args[1]
    target_index_list = unpickled_args[2]
    source_fname = unpickled_args[3]
    source_datasetpath = unpickled_args[4]
    source_index_list = unpickled_args[5]
    source_maskpath = unpickled_args[6]
    troi =  unpickled_args[7]
    
    verbose = unpickled_args[8]
    
    if len(unpickled_args)>9:
        timer_fname = unpickled_args[10]
    else:
        timer_fname = None


    slices = troi_to_slice(troi)
    
    if verbose:
        print('='*25)
        print('process {} is copying'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)
    
    if target_fname == source_fname:
        with h5py.File(target_fname) as h5_file:

            # import inspect 
            # linno = inspect.currentframe().f_lineno
            # print('DEBUG:\nin ' + __file__ + '\nline '+str(linno))
            # print('h5_file.keys()')
            # print(h5_file.keys())
            # print('source_datasetpath')
            # print(source_datasetpath)
            # print('target_datasetpath')
            # print(target_datasetpath)
            # print("h5_file['entry'].keys()")
            # print(h5_file['entry'].keys())
            # print("h5_file['entry/integrated'].keys()")
            # print(h5_file['entry/integrated'].keys())
            # print("h5_file['entry/integrated/troi1'].keys()")
            # print( h5_file['entry/integrated/troi1'].keys() )
            # print("h5_file['entry/integrated/troi1/raw_data'].keys()")
            # print(h5_file['entry/integrated/troi1/raw_data'].keys())


            mask = None
            if type(source_maskpath) != None:
                while type(mask) == type(None):
                    try:
                        mask = np.asarray(h5_file[source_maskpath][slices[0],slices[1]],dtype=np.bool)
                        mask = np.where(mask,0,1)
                    except IOError:
                        time.wait(0.001)
                        print('IO conflict process {} is waiting to read maskarray'.format(os.getpid()))
                    
                    
            for target_index, source_index in zip(target_index_list,source_index_list):

                if type(mask) != type(None):
                    source_data = h5_file[source_datasetpath][source_index][slices[0],slices[1]]*mask
                else:
                    source_data = h5_file[source_datasetpath][source_index][slices[0],slices[1]]
                                    
                h5_file[target_datasetpath][target_index] = source_data
                    
                if verbose:
                    print('process {} is copying frame {}'.format(os.getpid(),target_index))

            # h5_file.flush()
            
    else:
        with h5py.File(target_fname) as target_file:
            with h5py.File(source_fname) as source_file:
                for target_index, source_index in zip(target_index_list,source_index_list):
                    source_data = source_file[source_datasetpath][source_index][slices[0],slices[1]]
                    target_file[target_datasetpath][target_index] = source_data
                    if verbose:
                        print('process {} is copying frame {}'.format(os.getpid(),target_index))


            # target_file.flush()
            
    if verbose:
        print('process {} is done'.format(os.getpid()))
        print('='*25)

    
    if timer_fname != None:
        os.remove(timer_fname)


if __name__=='__main__':
    ''' 
    This is used by the lacal function copy_troi_employer(pickledargs_fname),
    DO NOT CHANGE
    '''
    if len(sys.argv)!=2:
        print('usage : python copy_troi_worker <pickled_instruction_list_fname>')
    pickledargs_fname = sys.argv[1]
    copy_troi_worker(pickledargs_fname)
                        
