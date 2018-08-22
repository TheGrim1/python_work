import time
import sys,os
import h5py
import numpy as np
import pyFAI
import fabio
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import fit_data_worker as fdw
import pythonmisc.pickle_utils as pu
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy


def do_fit(args):
    source_fname = args[0]
    dest_fname = args[1]
    verbose = args[2]
    fit_starttime = time.time()

    print('fitting diffraction data from {}'.format(dest_fname))
    
    index_min,index_max = (780, 1100)
    lorentz_index_guess = [13,29,245,264,130,142,88] # relative to the index range!
    
    poly_degree = 1
    
    source_grouppath = 'entry/integrated/q_radial/' 
    dest_grouppath = 'entry/fit/'

    no_lorentz = len(lorentz_index_guess)

    todo = [source_fname,
            source_grouppath,
            dest_fname,
            dest_grouppath,
            no_lorentz,
            lorentz_index_guess,
            poly_degree,
            (index_min,index_max),
            verbose]

    instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose)

    # print(instruction_fname)
    
    fdw.fit_data_employer(instruction_fname)
    
    fit_endtime = time.time()
    fit_time = (fit_endtime - fit_starttime)
    
    print('='*25)
    print('\n{}s taken for fitting \n{}'.format(fit_time, source_fname))
    print('='*25) 

    
