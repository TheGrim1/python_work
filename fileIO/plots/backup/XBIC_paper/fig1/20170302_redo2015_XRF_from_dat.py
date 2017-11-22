from __future__ import print_function
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


def crop_energyrange(data,energyrange):
    energystart = np.searchsorted(data[:,0], energyrange[0], 'left')
    energyend   = np.searchsorted(data[:,0], energyrange[1], 'right')
    return data[energystart:energyend,:]

def main():
    energyrange = [6.5,12.5]



    savepath = '/tmp_14_days/johannes1/results/mg01_5_4_3/singleXRF/'
    specpath = '/tmp_14_days/johannes1/results/mg01_5_4_3/singleXRF/xrf/'


    datadict = {}
    #getting data

    ### from the data files onga

    onga, ongaheader = open_data(specpath + 'onga.csv', delimiter = '\t',quotecharlist = ['"','#'])

    onga_counts = onga[:,[1,2]]
    onga_fit = onga[:,[1,3]]
    
    onga_counts = crop_energyrange(onga[:,[1,2]],energyrange)
    onga_fit = crop_energyrange(onga[:,[1,3]],energyrange)

    datadict.update({'onga_counts':onga_counts})
    datadict.update({'onga_fit':onga_fit})
    
    ### from the data files ongaas

    ongaas, ongaasheader = open_data(specpath + 'ongaas.csv', delimiter = '\t',quotecharlist = ['"','#'])

    ongaas_counts = ongaas[:,[1,2]]
    ongaas_fit = ongaas[:,[1,3]]
    
    ongaas_counts = crop_energyrange(ongaas[:,[1,2]],energyrange)
    ongaas_fit = crop_energyrange(ongaas[:,[1,3]],energyrange)


    datadict.update({'ongaas_counts':ongaas_counts})
    datadict.update({'ongaas_fit':ongaas_fit})
    
    ### from the data files onni
   

    onni, onniheader = open_data(specpath + 'onni.csv', delimiter = '\t',quotecharlist = ['"','#'])

    onni_counts = onni[:,[1,2]]
    onni_fit = onni[:,[1,3]]
    
    onni_counts = crop_energyrange(onni[:,[1,2]],energyrange)
    onni_fit = crop_energyrange(onni[:,[1,3]],energyrange)


    datadict.update({'onni_counts':onni_counts})
    datadict.update({'onni_fit':onni_fit})
    
    
    ### from the data files onwire
    
    onwire, onwireheader = open_data(specpath + 'onwire.csv', delimiter = '\t',quotecharlist = ['"','#'])

    onwire_counts = onwire[:,[1,2]]
    onwire_fit = onwire[:,[1,3]]
    
    onwire_counts = crop_energyrange(onwire[:,[1,2]],energyrange)
    onwire_fit = crop_energyrange(onwire[:,[1,3]],energyrange)

    
    datadict.update({'onwire_counts':onwire_counts})
    datadict.update({'onwire_fit':onwire_fit})

    fulldata, fullheader = combine_datasets(datadict)

    energy = np.atleast_1d(fulldata[:,0])

    fulldata = np.where(fulldata <= 1, 1, fulldata)
    ### plotting
    
    print(fulldata.shape)
    fig, axes = plt.subplots(nrows = 4, sharex = True)

    plt.rcParams.update({'font.size': 30})
    plotno = [5,7,3,1]
    color = ['blue','grey','green','red']
    for axno, dummy in enumerate(range(len(fulldata[1,:]))[1::2]):
        
        axes[axno].set_ylim([0.9,700])
        axes[axno].set_xlim([6.3,12.3])
        axes[axno].set_yscale('log')
        axes[axno].tick_params(length=8, width=2)
        axes[axno].vlines([7.46,8.26,9.23,10.26,10.51,11.72,9.71,11.44], ymin=0.9, ymax=700, colors=['blue','blue','red','red','green','green','black','black'])
        axes[axno].plot(energy,fulldata[:,plotno[axno]], color = color[axno], linewidth = 2)
        axes[axno].plot(energy,fulldata[:,plotno[axno]+1], color = 'black',linewidth = 2)
        


    plt.show()
        
        


    ### saving data
    save_data(savepath + 'single_spectra.dat', fulldata, fullheader, delimiter='\t')



if __name__ == "__main__":
    main()
