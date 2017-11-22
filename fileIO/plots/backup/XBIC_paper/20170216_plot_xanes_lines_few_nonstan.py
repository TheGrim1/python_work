from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
1
import sys,os
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.figsize': [11.0,10.0]})


import scipy.ndimage as nd
import matplotlib.colors as colors
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

from fileIO.datafiles.open_data import open_data



def main():


    ### reading data
    fname200 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_200ms_Ga/MG154_fluoXAS_1.replace.h5'
    fname400 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_400ms_Ga/MG154_fluoXAS_1.replace.h5'
    
    h5f200    = h5py.File(fname200,'r')

    ga200     = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/data'])[::-1,::-1,:]
    step200   = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/xanes_step'])[::-1,::-1]
    edge200   = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/xanes_edge'])[::-1,::-1]
    xbic200   = np.asarray(h5f200['/counters/zap_p201_Xbic_norm/data'])[::-1,::-1,:]
    mask200   = np.where(step200[:,:] > 5e-9,1,0)
    energy200 = np.asarray(h5f200['/detectorsum/Ga-K_norm_stan/energy'])

    h5f200.close()
    
    h5f400    = h5py.File(fname400,'r')   
    ga400     = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/data'])[::-1,::-1,:]
    step400   = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/xanes_step'])[::-1,::-1]
    edge400   = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/xanes_edge'])[::-1,::-1]
    mask400   = np.where(step400[:,:] > 12e-9,1,0)
    energy400 = np.asarray(h5f400['/detectorsum/Ga-K_norm_stan/energy'])

    h5f400.close()

    ### preparing what to plot:
    dataheader  = ['#energy [keV]']
    data1       = np.zeros(shape = (len(ga400[1,1,:]), np.sum(mask400) + 1))
    data1[:,0]  = energy400


#    plt.matshow(mask400)
#    plt.show()
    

    for i, y in enumerate(np.where(mask400)[0]):
        x = np.where(mask400)[1][i]
        dataheader.append([y,x])
        data1[:,i+1] = ga400[y,x,:] 
        
    ### smoothing the data
    for i in range(1,len(data1[1,:])):
        data1[:,i] = nd.gaussian_filter1d(data1[:,i],1)
        
  
        
    ### plotting

    energyrange = [10.360,10.390]
    
    energystart = np.searchsorted(energy400[:], energyrange[0], 'right')
    energyend   = np.searchsorted(energy400[:], energyrange[1], 'left')

    energy2d   = np.atleast_1d(energy400[energystart:energyend])
    data1      = data1[energystart:energyend,:]
    order      = [(y - x) for [y,x] in dataheader[1::]]

    order      = list(zip(order,list(range(1,len(order)+1))))

    dataordered= np.zeros(shape=data1.shape)
    order      = sorted(order)
    newheader  = [dataheader[0]]
   
    for i, (position, j) in enumerate(order):
        dataordered[:,i+1] = data1[:,j]
        newheader.append([dataheader[j],position])

    data1[:,1::] = dataordered[:,1::]
    dataheader = newheader
    positionlabel = ['position x=%s, y = %s, l = %s' % (x,y,l) for [[y,x],l] in dataheader[1::]]
    colorref = [old_div((l-0.5),10.0) for  [[y,x],l] in dataheader[1::]]
    cmap = plt.get_cmap('jet')
 
    fewdata      = np.zeros(shape = (data1.shape[0],11))
    fewdata[:,0] = data1[:,0]
    fewheader = [dataheader[0]] + list(range(10))
    
#    print(fewheader)
    fewnorm       = [0]*10
    fewcolorref   = [0]*10
#    print data1.shape
    for i in range(len(data1[1,1::])):
        a, position = dataheader[::1][i+1]
        fewnorm[position-1] +=1
        fewdata[:,position] += data1[:,i+1]
        fewcolorref[position-1] = colorref[i]
#        print position, fewnorm[position-1]


    for i, norm in enumerate(fewnorm):
        if norm !=0:
            fewdata[:,i+1] *= old_div(1.0,norm)
#        print fewdata[20,i+1]


  
    ### inset
    savename = '/tmp_14_days/johannes1/lincom/redo_xanes'
    insetshape = (max(np.where(mask400)[0])-min(np.where(mask400)[0])+1,
                  max(np.where(mask400)[1])-min(np.where(mask400)[1])+1)

    image = np.zeros(shape = insetshape)
    for [[y,x],l] in dataheader[1::]:
        image[y - min(np.where(mask400)[0]),
              x - min(np.where(mask400)[1])] = l

    fig, ax1 = plt.subplots()
    fig.set_size_inches(2,3)
    plt.matshow(image, cmap = cmap)
    fig.tight_layout()
    insetname = savename+'inset'
    plt.savefig(insetname + '.svg', transparent=True)
    plt.savefig(insetname + '.png', transparent=True)
    plt.savefig(insetname + '.eps', transparent=True)
    plt.savefig(insetname + '.pdf', transparent=True)
    
    plt.show()

        
    data1      = fewdata
    dataheader = fewdata
    colorref   = fewcolorref
    fig, ax1 = plt.subplots()
#    ax1.legend()
    for i in range(len(data1[1,1::])):
        ax1.plot(energy2d, data1[:,i+1]+old_div(i,10.0), 'b-',color = cmap(colorref[i]), linewidth = 2)

    xticks = list(range(int(energyrange[0]*1000),int(energyrange[1]*1000),5))
    ax1.set_xticks([old_div(x,1000.0) for x in xticks])
    ax1.set_xticklabels(['{:d}'.format(int(x * 1000)) for x in ax1.get_xticks()])
    ax1.set_xlabel('energy [eV]')
    ax1.set_ylabel('standardized Ga XRF signal')

    fig.tight_layout()

    plt.savefig(savename + '.svg', transparent=True)
    plt.savefig(savename + '.png', transparent=True)
    plt.savefig(savename + '.eps', transparent=True)
    plt.savefig(savename + '.pdf', transparent=True)
    plt.show()


    ### plot with lincom

    fig, ax1 = plt.subplots()
    fig.set_size_inches(5,5)
    lincomfname = '/tmp_14_days/johannes1/lincom/spectra/lin_components.dat'
    lincom, lincomheader = open_data(lincomfname,delimiter = '\t')
    lincom[:,0] = old_div(lincom[:,0],1000.0)-0.0025945
    energystart = np.searchsorted(lincom[:,0], energyrange[0], 'right')
    energyend   = np.searchsorted(lincom[:,0], energyrange[1], 'left')
    lincom       = lincom[energystart:energyend,:]
    lincomcolors = ['g','r','blue','darkblue']
    for i in range(1,len(lincom[0,:])):
        ax1.plot(lincom[:,0], lincom[:,i] + 1.5 +old_div(i,10.0), color = lincomcolors[i-1], linewidth = 2)
    ax1.legend(['                ']*4,loc=2)


    for i in range(len(data1[1,1::])):
        ax1.plot(energy2d, data1[:,i+1]+old_div(i,10.0), 'b-',color = cmap(colorref[i]), linewidth = 2)


    xticks = list(range(int(energyrange[0]*1000),int(energyrange[1]*1000),5))
    ax1.set_xticks([old_div(x,1000.0) for x in xticks])
    ax1.set_xticklabels(['{:d}'.format(int(x * 1000)) for x in ax1.get_xticks()])
    ax1.set_xlabel('energy [eV]')
    ax1.set_ylabel('standardized Ga XRF signal')

    fig.tight_layout()
    savename += '_with'
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


    f = open(savefile, 'w')
    f.writelines(savelines)
    f.close()

if __name__ == "__main__":
    main()
