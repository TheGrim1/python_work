'''
used to plot rois from the increasing Temperature KMAPs
'''

import sys, os
import glob
import numpy as np
import matplotlib.pyplot as plt
from silx.io.spech5 import SpecH5

sys.path.append('/data/id13/inhouse2/AJ/skript')
from fileIO.edf.save_edf import save_edf

def plot_roi(all_data, roiname, temp):

    save_path = '/data/id13/inhouse2/AJ/data/ma3576/id01/analysis/fluence/T_{}/'.format(temp)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_fname_tpl = save_path + 'T_{}_{}.{}' # .format(T,roiname,file_type)

    fig, ax = plt.subplots(1,6)

    vmin = all_data.min()
    vmax = all_data.max()
    # vmin = np.percentile(all_data,1)
    # vmax = np.percentile(all_data,99)
    
    for i, dataset in enumerate(all_data):
        ax[i].imshow(dataset,vmin=vmin,vmax=vmax)
        ax[i].yaxis.set_ticklabels([])
        ax[i].xaxis.set_ticklabels([])
        ax[i].set_yticks([])
        ax[i].set_xticks([])

        save_edf(dataset, save_fname_tpl.format(temp,roiname+'_'+str(i),'edf'))

    plt.savefig(save_fname_tpl.format(temp,roiname,'png'), transparent=True)
                

    
def main():

    FNAME_LIST= glob.glob('/mntdirect/_data_id13_inhouse2/AJ/data/ma3576/id01/metadata/ma3576/id01/fluence/**/spec/**_fast_**')

    exclude = ['084420','072141']
    FNAME_LIST = [x for x in FNAME_LIST if not any([x.find(y)>0 for y in exclude])]
    FNAME_LIST.sort()
    T_list = [30,35,40,45,50,55,60,65,70,70,75,80]
    FNAME_LIST = FNAME_LIST[:len(T_list)]

    roiname_list = [u'mpx4int',
                    u'mpx4ro1',
                    u'mpx4ro2',
                    u'roi1',
                    u'roi2',
                    u'roi3',
                    u'roi4',
                    u'roi5']


    
    for fname, temp in zip(FNAME_LIST, T_list):
        print('on: ' + fname)
        sfh5 = SpecH5(fname)
        for roiname in roiname_list:
            print(roiname)
            all_data = np.zeros(shape = (6,130,85))
            for i, scanno in enumerate(sfh5.keys()):
                print(scanno)
                dataset = np.asarray(sfh5[scanno]['measurement'][roiname])
                dataset = np.rollaxis(dataset.reshape(len(dataset)/130,130),-1)[::-1,:]
                all_data[i][:dataset.shape[0],:dataset.shape[1]] += dataset

            plot_roi(all_data,roiname,temp)
            
        
    

if __name__=='__main__':
    main()
