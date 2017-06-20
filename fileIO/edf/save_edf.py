# global imports
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import fabio
import subprocess

# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice

def save_edf(data, filename, threshold = 0.0):
    '''creates or overwritees filename!\nThen saves data as filename (presumably .edf)\n DOES NOT WORK'''

    
    if filename.find(".edf") != -1:   
        subprocess.call(['touch',filename])
        edffile = fabio.open(filename)        

        if threshold != 0.0:
            data  = np.where(data < threshold, data, 0)
        
        edffile.data = data
        
        return data
    else:
        print "%s is not a .edf file?" %filename
