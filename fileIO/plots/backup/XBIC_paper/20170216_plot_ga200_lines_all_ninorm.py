
import sys, os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import matplotlib.colors as colors

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice
from fileIO.plots.plot_tools import draw_lines_troi
from simplecalc.calc import normalize_self

def main():


    ### reading data
    fname200 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_200ms_Ga/MG154_fluoXAS_1.replace.h5'
    fname400 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_400ms_Ga/MG154_fluoXAS_1.replace.h5'

    h5f200    = h5py.File(fname200,'r')

    ni200 = np.asarray(h5f200['detectorsum/Ni-K/data'])[::-1,::-1,:]
    ni200mask  = np.zeros(shape=ni200[:,:,1].shape)
    ni200troi   = ((2,10),(6,5))
    ni200mask[troi_to_slice(ni200troi)] = 1
    norm = np.zeros(shape = (ni200.shape[2]))
    for i, y in enumerate(np.where(ni200mask)[0]):
        x = np.where(ni200mask)[1][i]
        norm += ni200[y,x,:]
    norm = norm/np.max(norm)
    norm = np.atleast_1d(norm)


    
    ga200     = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/data'])[::-1,::-1,:]

    ga200_norm= np.asarray(h5f200['/detectorsum/Ga-K_norm/data'])[::-1,::-1,:]
    ga200_ninorm = np.asarray(h5f200['/detectorsum/Ga-K/data'])[::-1,::-1,:] / np.where(norm >0, norm, 1)
    
    step200   = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/xanes_step'])[::-1,::-1]
    edge200   = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/xanes_edge'])[::-1,::-1]
    xbic200   = np.asarray(h5f200['/counters/zap_p201_Xbic_norm_stan/data'])[::-1,::-1,:]
    xbic200_norm   = np.asarray(h5f200['/counters/zap_p201_Xbic_norm/data'])[::-1,::-1,:]
    xbic200_step   = np.asarray(h5f200['/counters/zap_p201_Xbic_norm_stan/xanes_step'])[::-1,::-1]
    mask200   = np.where(step200[:,:] > 5e-9,1,0)
    energy200 = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/energy'])

    plt.plot(energy200,norm)
    plt.show()
    
    xbicmask  = np.zeros(shape=xbic200[:,:,1].shape)
    xbictroi   = ((9,6),(2,2))
    xbicmask[troi_to_slice(xbictroi)] = 1
             
    h5f200.close()
    
    h5f400    = h5py.File(fname400,'r')   
    ga400     = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/data'])[::-1,::-1,:]
    step400   = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/xanes_step'])[::-1,::-1]
    edge400   = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/xanes_edge'])[::-1,::-1]
    mask400   = np.where(step400[:,:] > 12e-9,1,0)
    energy400 = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/energy'])

    h5f400.close()

    ### preparing what to plot:
    dataheader  = ['energy [keV]']
    data1       = np.zeros(shape = (len(ga200_norm[1,1,:]), np.sum(xbicmask) + 1))
    data1[:,0]  = energy200


    for i, y in enumerate(np.where(xbicmask)[0]):
        x = np.where(xbicmask)[1][i]
        dataheader.append([y,x])
        data1[:,i+1] = ga200_ninorm[y,x,:] 

    ### smoothing the data
#    for i in range(1,len(data1[1,:])):
#        data1[:,i] = nd.gaussian_filter1d(data1[:,i],1)
        
        
    ### plotting
    fig, ax1 = plt.subplots()

    energyrange = [10.350,10.420]
    
    energystart = np.searchsorted(energy400[:], energyrange[0], 'right')
    energyend   = np.searchsorted(energy400[:], energyrange[1], 'left')

    energy2d   = np.atleast_1d(energy400[energystart:energyend])
    data1      = data1[energystart:energyend,:]
    order      = [(y - x) for [y,x] in dataheader[1::]]

    order      = zip(order,range(1,len(order)+1))

    dataordered= np.zeros(shape=data1.shape)
    order      = sorted(order)
    newheader  = [dataheader[0]]
   
    for i, (position, j) in enumerate(order):
        dataordered[:,i+1] = data1[:,j]
        newheader.append([dataheader[j],position])

    data1[:,1::] = dataordered[:,1::]
    dataheader = newheader
    positionlabel = ['position x=%s, y = %s, l = %s' % (x,y,l) for [[y,x],l] in dataheader[1::]]
    colorref = [(l-0.5)/10.0 for  [[y,x],l] in dataheader[1::]]
    cmap = plt.get_cmap('coolwarm')
   



    cycler =  ['r', 'g', 'b', 'y']
    for i in range(len(data1[1,1::])):
        ax1.plot(energy2d, data1[:,i+1], color = cycler[i], linewidth = 2)

#    xticks = range(10360,10390,5)
#    ax1.set_xticks([x/1000.0 for x in xticks])
    ax1.set_xticklabels(['{:d}'.format(int(x * 1000)) for x in ax1.get_xticks()])
    ax1.set_xlabel('energy [eV]')
    ax1.set_ylabel('Ga-K XRF signal normalized by Ni signal')

    fig.tight_layout()
    savename = '/tmp_14_days/johannes1/lincom/spectra/redo_ga_ninorm'
    plt.savefig(savename + '.svg', transparent=True)
    plt.savefig(savename + '.png', transparent=True)
    plt.savefig(savename + '.eps', transparent=True)
    plt.savefig(savename + '.pdf', transparent=True)
    plt.show()




    ### savedata
    savefile = savename + '.dat'
    savelines = ['\t'.join(['#energy [kev]']+ [str(x) for x in range(10)] + ['\n'])]
    for line in data1:
        savelines.append('\t'.join([str(x) for x in line] + ['\n']))

    print savelines
    f = open(savefile, 'w')
    f.writelines(savelines)
    f.close()



    extent = 0 , 18, 34, 0
    im1 = plt.matshow(np.sum(ga200_norm[:,:,:],axis = -1), cmap = plt.cm.gnuplot2, fignum = 0, extent = extent)
    
    draw_lines_troi(xbictroi, color = 'black', axes = im1.axes, linewidth = 3)
    savename = savename + '_map'
    plt.savefig(savename + '.svg', transparent=True)
    plt.savefig(savename + '.png', transparent=True)
    plt.savefig(savename + '.eps', transparent=True)
    plt.savefig(savename + '.pdf', transparent=True)
    plt.show()
    
    

    
if __name__ == "__main__":
    main()
