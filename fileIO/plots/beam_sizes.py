from __future__ import print_function
from __future__ import division
from past.utils import old_div
import numpy as np
import matplotlib.pyplot as plt

import fileIO.datafiles.open_data as od
import os,sys
import simplecalc.fitting as fit
import simplecalc.calc as calc
import scipy.ndimage as nd

def plot_beamsize_to_ax(data,ax):
    ax.set_xlim(-380,220)
    ax.set_ylim(-0.1,1.1)
    data=data[np.where(data[:,0]>-300.5)]
    data=data[np.where(data[:,0]<300.5)]
    ax.plot(data[:,0],data[:,1],'bo')
    beta = fit.do_gauss_fit(data,verbose=False)
    ax.plot((np.arange(600)-380), fit.gauss_func(beta, (np.arange(600)-380)), "r--", lw = 2)
    print('FWHM = ' , calc.get_fwhm(data))
    print('sigma = ', beta[2])


def plot_beamsize(data):
    fig, ax = plt.subplots()
    plt.plot(data[:,0],data[:,1])
    beta = fit.do_gauss_fit(data,verbose=False)
    ax.plot(data[:,0], fit.gauss_func(beta, data[:,0]), "r--", lw = 2, label = 'data' )
    print('FWHM = ' , calc.get_fwhm(data))
    print('sigma = ', beta[2])
    plt.show()
    
def normalize_scan_data(data):
    '''
    data[:,0] -> x is reduced to the intervallshifted so that the max is in the middle
    data[:,1] is normalized to span [0,1]
    '''

    data[:,1] += -np.min(data[:,1])
    data[:,1] *= old_div(1.0,np.max(data[:,1]))
    data[:,0] += -data[:,0][np.where(data[:,1]==np.max(data[:,1]))]
    
    return data

def _get_data(fname):
    
    data = od.open_data(fname)[0]
    data = normalize_scan_data(data)
    plot_beamsize(data)
