from __future__ import print_function
from __future__ import division

from past.utils import old_div
import sys, os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import matplotlib.colors as colors

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice
from simplecalc.calc import normalize_self
from fileIO.plots.plot_tools import draw_lines_troi



def main():


    ### reading data
    fname200 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_200ms_Ga/MG154_fluoXAS_1.replace.h5'
    fname400 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_400ms_Ga/MG154_fluoXAS_1.replace.h5'
    
    h5f200    = h5py.File(fname200,'r')

    ni200 = np.asarray(h5f200['detectorsum/Ni-K/data'])
    xbic  =  np.asarray(h5f200['counters/zap_p201_Xbic/data'])
    xbic_norm =  np.asarray(h5f200['counters/zap_p201_Xbic_norm/data'])
    norm      = old_div(xbic, np.where(xbic_norm >0, xbic_norm, 1))
    energy200 = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/energy'])

    norm = normalize_self(norm)
    ni200_norm = old_div(ni200, np.where(norm >0, norm, 1))
    
    ni200mask  = np.zeros(shape=ni200[:,:,1].shape)
    ni200troi   = ((2,10),(6,5))
    ni200mask[troi_to_slice(ni200troi)] = 1
             
    h5f200.close()
    
    h5f400    = h5py.File(fname400,'r')
    ni400 = np.asarray(h5f400['detectorsum/Ni-K/data'])

    energy400 = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/energy'])
    
    
    ni400mask  = np.zeros(shape=ni400[:,:,1].shape)
    ni400troi   = ((3,12),(3,3))
    ni400mask[troi_to_slice(ni400troi)] = 1
             
    h5f400.close()

    ### preparing what to plot:
    dataheader  = ['energy [keV]']
    data1       = np.zeros(shape = (len(ni200[1,1,:]), np.sum(ni200mask) + 1))
    data1[:,0]  = energy200
    data2 = np.copy(data1)
    data3 = np.copy(data1)

    for i, y in enumerate(np.where(ni200mask)[0]):
        x = np.where(ni200mask)[1][i]
        dataheader.append([y,x])
        data1[:,i+1] = ni200[y,x,:]
        data2[:,i+1] = ni200[5,3,:]
        data3[:,i+1] = ni200[6,6,:]
        

    ### smoothing the data
#    for i in range(1,len(data1[1,:])):
#        data1[:,i] = nd.gaussian_filter1d(data1[:,i],1)


    sumni      = np.zeros(shape = (data1.shape[0],4))
    sumni[:,0] = data1[:,0]
    sumni[:,1] = np.sum(data1[:,1::],axis = -1)
    sumni[:,2] = np.sum(data2[:,1::],axis = -1)
    sumni[:,3] = np.sum(data3[:,1::],axis = -1)

    

                          
    ### plotting
    fig, ax1 = plt.subplots()

    energyrange = [10.350,10.420]
    
    energystart = np.searchsorted(energy400[:], energyrange[0], 'right')
    energyend   = np.searchsorted(energy400[:], energyrange[1], 'left')

    energy2d   = np.atleast_1d(energy400[energystart:energyend])
    sumni      = sumni[energystart:energyend,:]
   


    cycler =  ['r', 'g', 'b', 'y']
    for i in range(len(sumni[1,1::])):
        ax1.plot(energy2d, sumni[:,i+1], color = cycler[i], linewidth = 2)

#    xticks = range(10360,10390,5)
#    ax1.set_xticks([x/1000.0 for x in xticks])
    ax1.set_xticklabels(['{:d}'.format(int(x * 1000)) for x in ax1.get_xticks()])
    ax1.set_xlabel('energy [eV]')
    ax1.set_ylabel('non standardized Ni signal')

    fig.tight_layout()
    savename = '/tmp_14_days/johannes1/lincom/spectra/check_ni'
    plt.savefig(savename + '.svg', transparent=True)
    plt.savefig(savename + '.png', transparent=True)
    plt.savefig(savename + '.eps', transparent=True)
    plt.savefig(savename + '.pdf', transparent=True)
    plt.show()

# test how Ni correctrion effects Ni signal
    fig, ax1 = plt.subplots()
    dummy = np.copy(sumni[:,1])
    sumni[:,1] = normalize_self(sumni[:,1])
    sumni[:,2] = old_div(sumni[:,2],dummy)
    sumni[:,3] = old_div(sumni[:,3],dummy)


    for i in range(len(sumni[1,1::])):
         ax1.plot(energy2d, sumni[:,i+1], color = cycler[i], linewidth = 2)
    plt.show()
    


    ### savedata
    savefile = savename + '.dat'
    savelines = ['\t'.join(['#energy [kev]']+ [str(x) for x in range(10)] + ['\n'])]
    for line in sumni:
        savelines.append('\t'.join([str(x) for x in line] + ['\n']))

    print(savelines)
    f = open(savefile, 'w')
    f.writelines(savelines)
    f.close()

    
    extent = 0 , 18, 34, 0
    im1 = plt.matshow(np.sum(ni200[:,:,:],axis = -1), cmap = plt.cm.gnuplot2, fignum = 0, extent = extent)
    
    draw_lines_troi(ni200troi, color = 'black', axes = im1.axes, linewidth = 3)
    savename = savename + '_map'
    plt.savefig(savename + '.svg', transparent=True)
    plt.savefig(savename + '.png', transparent=True)
    plt.savefig(savename + '.eps', transparent=True)
    plt.savefig(savename + '.pdf', transparent=True)
    plt.show()
    
    

    
if __name__ == "__main__":
    main()
