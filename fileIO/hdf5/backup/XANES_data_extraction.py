

from builtins import str
from builtins import range
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import h5py
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.calc import normalize_self
from simplecalc.calc import normalize_xanes
def main():


    ### get data ###    
    fname200 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_200ms_Ga/MG154_fluoXAS_1.replace.h5'
    h5f = h5py.File(fname200,"r")
    xbic200 = np.asarray(h5f['counters/zap_p201_Xbic_norm/data'])
    ga200 = np.asarray(h5f['detectorsum/Ga-K_norm/data'])
    energy = h5f['axes/energy/data']

    fname400 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_400ms_Ga/MG154_fluoXAS_1.replace.h5'
    h5f400 = h5py.File(fname400,"r")
    ga400 = np.asarray(h5f400['detectorsum/Ga-K_norm/data'])
    energy400 = h5f400['axes/energy/data']
    
    ### normalization ###
    
    xp1 = np.copy(xbic200[23,10,:])
    xp1 = normalize_self(xp1)

    xp2 = np.copy(xbic200[25,9,:])
    xp2 = normalize_self(xp2)

    gap1 = np.copy(ga200[23,10,:])
    gap1 = normalize_self(gap1)

    gap2 = np.copy(ga200[25,9,:])
    gap2 = normalize_self(gap2)

    gap400_1 = np.copy(ga400[37,17,:])
    gap400_2 = np.copy(ga400[40,15,:])
    
    gap400_2 = normalize_self(gap400_2)
    gap400_1 = normalize_self(gap400_1)



    ### plotting ####
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)

    ax1.set_title('red - interface, blue - "GaAs"')
    
    ax1.plot(energy, xp1, 'red')
    ax1.plot(energy, xp2 , 'blue')

    ax2.plot(energy, gap1 , 'red')
    ax2.plot(energy, gap2 , 'blue')

    ax3.plot(energy400, gap400_1 , 'red')
    ax3.plot(energy400, gap400_2 , 'blue')

    
    
    ax1.set_ylabel('XBIC signal')
    ax2.set_ylabel('Ga-K XRF 200 ms')
    ax3.set_ylabel('Ga-K XRF 400 ms')

    ax3.set_xlabel('Energy [keV]')


    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    # fig, ax1 = plt.subplots()   

    
    # ax1.set_xlabel('map no -> energy')
    
    # ax1.set_ylabel('XBIC signal', color='b')
    # ax1.tick_params('y', colors='b')

    # ax2 = ax1.twinx()

    # ax2.set_ylabel('GaKa Sum', color='r')
    # ax2.tick_params('y', colors='r')
    
#   f.tight_layout()
    plt.show()


    ### saving the data

    savefile = '/tmp_14_days/johannes1/XANES_single_points_200ms.dat'

    f= open(savefile,"w+")

    datatowrite = np.vstack([energy, xp1, xp2, gap1, gap2])


    lines = ['\t'.join(['energy','XBIC_23_10','XBIC_25_9','GaK_23_10','GaK_25_9'])+'\n']

    for i in range(len(datatowrite[0])):
        line = []
        for k in range(len(datatowrite)):
            line.append(str(datatowrite[k][i]))
        lines.append('\t'.join(line)+'\n')

    f.writelines(lines)
                       
    f.close()
    
    savefile = '/tmp_14_days/johannes1/XANES_single_points_400ms.dat'

    f= open(savefile,"w+")

    datatowrite = np.vstack([energy400, gap400_1, gap400_2])


    lines = ['\t'.join(['energy','GaK_37_17','GaK_40_15'])+'\n']

    for i in range(len(datatowrite[0])):
        line = []
        for k in range(len(datatowrite)):
            line.append(str(datatowrite[k][i]))
        lines.append('\t'.join(line)+'\n')

    f.writelines(lines)
                       
    f.close()
    
    
    
if __name__ == '__main__':
  
    main()




group = 'detectorsum/Ga-K_norm',
e0 = 10.367
preedge = (0,10.352)
postedge = (10.392, 10.427)
fitorder = 1
import h5py
h5f = h5py.File(fname200,'r')
fname200 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_200ms_Ga/MG154_fluoXAS_1.replace.h5'
h5f = h5py.File(fname200,'r')
import numpy as np
import fileIO.hdf5.h5_xanestools as xanes
import simplecalc.calc as calc
data = h5f['/counters/zap_p201_Xbic_norm/data']
energy = h5f['/counters/zap_p201_Xbic_norm/energy']

calc.normalize_xanes(np.transpose(np.asarray([energy,data[23,10,:]])), e0 = e0, preedge = preedge, postedge = postedge, verbose = 4)
