from __future__ import division
from past.utils import old_div
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'figure.figsize': [4.0,6.0]})

import sys, os

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

from fileIO.datafiles.open_data import open_data
from fileIO.datafiles.save_data import save_data
from simplecalc.calc import normalize_xanes, combine_datasets
from simplecalc.linear_combination import do_component_analysis
import numpy as np


def crop_posrange(data,posrange):
    posstart = np.searchsorted(data[:,0], posrange[0], 'left')
    posend   = np.searchsorted(data[:,0], posrange[1], 'right')
    return data[posstart:posend,:]

def main():
    posrange = [0.5,3.5]

    savepath = '/tmp_14_days/johannes1/results/mg01_5_4_3/singleXRF/'
    specpath = '/tmp_14_days/johannes1/results/mg01_5_4_3/singleXRF/line/'

    data, header = open_data(specpath + 'together.dat', delimiter = '\t')

    data[:,1::] = old_div(data[:,1::],np.max(data[:,2]))

    data = crop_posrange(data,posrange)
    data[:,3] = old_div(data[:,3],np.max(data[:,3]))
    pos = np.atleast_1d(data[:,0]-data[0,0])

    color = ['red','green','blue']
    fig, ax1 = plt.subplots()
    
    for i in range(1,len(data[1,:])):
        plt.rcParams.update({'font.size': 20})

        ax1.set_yscale('log')
        ax1.set_ylim([0.03,1.2])
        ax1.plot(pos,data[:,i],color[i-1],linewidth=2)
        ax1.set_yticks([1,0.5,0.1])
        ax1.set_yticklabels(['1','0.5','0.1'])

    ax1.legend(['Ga/(Ga+As)','(Ga+As)','Ni'],loc = 2)
        
    plt.show()



    ### saving data
    save_data(savepath + 'linescan.dat', data, header, delimiter='\t')



if __name__ == "__main__":
    main()
