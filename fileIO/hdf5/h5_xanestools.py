from __future__ import print_function
import numpy as np
import sys, os
import matplotlib.pyplot as plt
import h5py
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.calc import normalize_xanes

def xanes_standardize(h5fname,
                      group = 'detectorsum/Ga-K_norm',
                      e0 = 10.367,
                      preedge = (0,10.352), 
                      postedge = (10.392, 10.427),
                      fitorder = 1,
                      verbose = False):
    '''
    goes into h5fname[group/data], expecting Y (x Z) x Energy data with <h5fname>[<group>/energy] as the energy
    rund simplecalc.calc.normalize_xanes on each spectrum in that dataset using e0, preedge, postedge, fitorder=1
    saves the result as data in <h5fname> as <group + _stan>, setting a link to the other keys.
    adds to <group + _stan> /xanes_edge and /xanes_step as Y (x Z) arrays
    '''

    ### reading data
    h5f    = h5py.File(h5fname,"r+")
    data   = np.asarray(h5f[group + '/data'])
    energy = np.asarray(h5f[group + '/energy'])

    

    # ### correcting the energy scale:
    # def energy_correction(x):
    #     measuredvalue = 10.36874710506343
    #     shouldbevalue = 10.365698697738177
    #     return x + (shouldbevalue - measuredvalue)

    # energy = energy_correction(energy)
            

    ## fitting etc.

    data_stan = np.zeros(shape=data.shape) 


    if len(data.shape) == 2:
        edge = np.zeros(shape = data.shape[0])
        step = np.zeros(shape = data.shape[0])
        dataplaceholder = np.empty(shape = (data.shape[0],2))
        for z in range(data.shape[0]):        
            (dataplaceholder , data_stan[z,:],edge[z],step[z]) =  normalize_xanes(np.transpose(np.asarray([energy,data[z,:]])),
                                                                                   e0,
                                                                                   preedge,
                                                                                   postedge,
                                                                                   fitorder = fitorder,
                                                                                   verbose = verbose)
            data_stan[z,:] = np.copy(data_placeholder[:,1])
    elif len(data.shape) == 3:
        edge = np.zeros(shape = (data.shape[0],data.shape[1]))
        step = np.zeros(shape = (data.shape[0],data.shape[1]))
        dataplaceholder = np.empty(shape = (data.shape[0],2))
        for z in range(data.shape[0]):
            for y in range(data.shape[1]):
                (dataplaceholder, edge[z, y], step[z, y]) =  normalize_xanes(np.transpose(np.asarray([energy,data[z,y,:]])),
                                                                                           e0,
                                                                                           preedge,
                                                                                           postedge,
                                                                                           fitorder = fitorder,
                                                                                           verbose = verbose)
                data_stan[z, y, :] = np.copy(dataplaceholder[:,1])
                
    else:
        raise IndexError('data dimension %s not implemented' % data.dim)
                

    
    ### saving
    
    savegroup = h5f.create_group(group + '_stan')
    
    ### cloning the previos group except for data
    
    for name, dataset in list(h5f[group].items()):
        if not name.find('data') and not name.find('energy'):
            savegroup.create_dataset(name, data = dataset)

    ### saving the standardized data
    savegroup.create_dataset('data', data = data_stan)
    savegroup.create_dataset('energy', data = energy)
    savegroup.create_dataset('xanes_edge', data = edge)
    savegroup.create_dataset('xanes_step', data = step)

    
    h5f.close()        
    
    
   

def main(args):
    print(args)
    fname = args[0]
    if len(args) >1:
        group = args[1]
    else:
        group = 'detectorsum/Ga-K_norm'
    if fname.find(".h5"):
        xanes_standardize(fname, group = group)
                
if __name__ == '__main__':
    
    args = []
    if len(sys.argv) > 1:
        if sys.argv[1].find("-f")!= -1:
            f = open(sys.argv[1]) 
            for line in f:
                args.append(line.rstrip())
        else:
            args=sys.argv[1:]
    else:
        f = sys.stdin
        for line in f:
            args.append(line.rstrip())
    
#    print args
    main(args)

def stugff():
    fname200 = '/tmp_14_days/johannes1/results/MG154_fluoXAS_1_200ms_Ga/MG154_fluoXAS_1.replace.h5'
    f200 = h5py.File(fname,'r')
    list(f200.items())
