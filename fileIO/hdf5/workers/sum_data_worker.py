from __future__ import print_function
import numpy as np
import h5py
import os

def sum_data_worker(inargs):
    '''
    copies troi into target_fname[target_datasetpath][target_index] from source_name[source_datasetpath][source_index][troi]
    these dataset have allready exist with the right shape and dtype!
    '''
    
    target_fname = inargs[0]
    target_datasetpath = inargs[1]
    sum_indexes = inargs[2]
    verbose = inargs[3]

       
    if verbose:
        print('='*25)
        print('process {} is summing up'.format(os.getpid()))
        for arg in inargs:
            print(arg)

    data_sum = []
    with h5py.File(target_fname) as h5_file:
        for frame in sum_indexes:
            source_data = h5_file[target_datasetpath][frame]
            frame_sum = np.sum(source_data)
            #print(source_data)
            # print('{} {}'.format(frame,frame_sum))
            # print('source_data.shape')
            # print(source_data.shape)
            # print('np.max(source_data)')
            # print(np.max(source_data))
            if verbose:
                print('process {} is summing frame {}'.format(os.getpid(),frame))
            data_sum.append([frame,frame_sum])
    
    
    return data_sum
