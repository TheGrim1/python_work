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

def crop_energyrange(data,energyrange):
    energystart = np.searchsorted(data[:,0], energyrange[0], 'left')
    energyend   = np.searchsorted(data[:,0], energyrange[1], 'right')
    return data[energystart:energyend,:]



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

    energyrange = [10.340,10.400]

    data1 = crop_energyrange(data1,energyrange)
    
    # energystart = np.searchsorted(energy400[:], energyrange[0], 'right')
    # energyend   = np.searchsorted(energy400[:], energyrange[1], 'left')
    # data1       = data1[energystart:energyend,:]

    energystart = np.searchsorted(energy400[:], energyrange[0], 'right')
    energyend   = np.searchsorted(energy400[:], energyrange[1], 'left')
    energy2d   = energy400[energystart:energyend]

    
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
    cmap = plt.get_cmap('jet')
 
    fewdata      = np.zeros(shape = (data1.shape[0],11))
    fewdata[:,0] = data1[:,0]
    fewheader = [dataheader[0]] + range(10)
    
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
            fewdata[:,i+1] *= 1.0/norm
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
        ax1.plot(energy2d, data1[:,i+1]+i/10.0, 'b-',color = cmap(colorref[i]), linewidth = 2)


    xticks = range(int(energyrange[0]*1000),int(energyrange[1]*1000),5)
    ax1.set_xticks([x/1000.0 for x in xticks])
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
    print lincomheader
    
    lincom[:,0] = lincom[:,0]/1000.0
    energystart = np.searchsorted(lincom[:,0], energyrange[0], 'right')
    energyend   = np.searchsorted(lincom[:,0], energyrange[1], 'left')
    lincom       = lincom[energystart:energyend,:]
    lincomcolors = ['g','r','blue','darkblue']
    for i in [2,3,4,5]:
        ax1.plot(lincom[:,0], lincom[:,i] + 1.5 +i/10.0, color = lincomcolors[i-2], linewidth = 2)
    ax1.legend(['GaAs','Ga-metal','alpha-Ga2O3','beta-Ga2O3'],loc=2)


    ### measured_correction can't be calculated live or saved in the data because the results may be circularily used!
    measured_correction = 2.59453867829 / 1000
    
    for i in range(len(data1[1,1::])):
        ax1.plot(energy2d + measured_correction, data1[:,i+1]+i/10.0, 'b-',color = cmap(colorref[i]), linewidth = 2)


    ### start add indication of edge shift
    e0_gaas  = np.interp(0.5, lincom[:,2], lincom[:,0])
    e0_meas  = []
    for i in range(len(data1[1,1::])):
        e0_meas.append(np.interp(0.5, data1[:,i+1], energy2d + measured_correction))
    print e0_meas

    ax1.vlines(e0_gaas,0,2.2, colors = 'green', linewidth = 2)
    
    ax1.hlines(np.arange(0.5,1.5,0.1),e0_gaas,e0_meas[:],colors='black',linewidth = 2)
    
    e0_meas = np.asarray(e0_meas)- e0_gaas
    print e0_meas

    ### end add indication of edge shift

    energyrange = [10.360, 10.390]
    ax1.set_xlim(energyrange[0],energyrange[1])    
    ax1.set_ylim(0,4.5)
    xticks = range(int(energyrange[0]*1000),int(energyrange[1]*1000),5)
    ax1.set_xticks([x/1000.0 for x in xticks])
    ax1.set_xticklabels(['{:d}'.format(int(x * 1000)) for x in ax1.get_xticks()])
    ax1.set_xlabel('energy [eV]')
    ax1.set_ylabel('standardized and offset Ga XRF signal')

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

    ### plot edge_shift
    ax1 = plt.gca()

    colorlist = []
    for i in range(len(e0_meas)):
        colorlist.append(cmap(colorref[i]))

    ax1.hlines(0,0,12,linewidth = 1,color = 'black')
    ax1.bar(np.asarray(range(len(e0_meas)))+0.5, e0_meas*1000, 0.8, color = colorlist)
    xticks = range(1,11)
    ax1.set_xticks([x for x in xticks])
    ax1.xaxis.set_ticks_position('top')
    ax1.set_xlim(0.1,10.9)
    ax1.set_yticks([0,-0.5,-1])
    ax1.set_xticklabels(['{:d}'.format(int(x)) for x in ax1.get_xticks()])


    ax1.set_ylabel('shift of absorption edge [eV]')
    plt.show()

    

    ### end plot edge_shift

    
if __name__ == "__main__":
    main()
