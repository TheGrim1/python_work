from __future__ import print_function

import pickle 
import numpy as np
import os
import time

def pickle_list(the_list):
    '''
    individually pickles every item in list, returns pickled list
    '''    
    pickled_list = [pickle.dumps(item) for item in the_list]
    
    return pickled_list

def unpickle_list(pickled_list):
    '''
    individually unpickles every item in list, returns unpickled list
    '''

    unpickled_list = [pickle.loads(item) for item in pickled_list]
    
    return unpickled_list

def pickle_to_file(to_pickle, verbose = False, caller_id=0, basepath = '/tmp/',counter=None):

    if counter == None:
        np.random.seed(int(time.time()))
        counter = np.random.randint(1000000)
    fname = os.path.sep.join([basepath,'delete_me_{}_{}.tmp'.format(caller_id, counter)])
    if verbose:
        print('pickling to {}'.format(fname))

    f = open(fname,'w')
    pickled = pickle.dumps(to_pickle)
    f.write(pickled)
        
    f.flush()
    f.close()

    return fname

def unpickle_from_file(fname, delete = True, verbose = False):
    if verbose:
        print('unpickling list from {}'.format(fname))
    f = open(fname,'r')
    
    unpickled = pickle.loads(f.read())

    f.close()
    if delete:
        os.remove(fname)
    return unpickled
