
import sys, os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import matplotlib.colors as colors

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice
from fileIO.plots.plot_tools import draw_lines_troi

def main():

    ### reading data
    fname2015 = '/tmp_14_days/johannes1/mg01_5_4_3/results/mg01_5_4_3/mg01_5_4_3.replace.h5'
    h5f2015   = h5py.File(fname2015,'r')
