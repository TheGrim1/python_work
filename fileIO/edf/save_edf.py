from __future__ import print_function
# global imports
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import fabio
import subprocess

def save_edf(data, filename, threshold = 0.0):
    '''creates or overwritees filename!\nThen saves data as filename (presumably .edf)'''

    
    if filename.find(".edf") != -1:   
        subprocess.call(['touch',filename])
        edffile = fabio.edfimage.EdfImage()

        if threshold != 0.0:
            data  = np.where(data < threshold, data, 0)
        
        edffile.data = data
        edffile.write(filename)
        return data
    else:
        print("%s is not a .edf file?" %filename)