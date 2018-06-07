import sys,os
import numpy as np
import h5py
from multiprocessing import Pool
import matplotlib.pyplot as plt


sys.path.append('/data/id13/inhouse2/AJ/skript/')
import fileIO.images.image_tools as it


def main():
    
    merged_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_gpu2/merged.h5'
    savepath = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_gpu2/peaks_pos/'

    troi_list =['troi_ml','troi_tr']
    tth_troi_list = [[[174.726,10.3747],[43.4175,5.99692]],[[219.882,13.5378],[31.07,6.31762]]]
    g_path_tpl = 'entry/merged_data/diffraction/{}/single_maps'
    savename_tpl = 'peakpos_{}_{}_{:06d}.png'

    data_path_list = []
    with h5py.File(merged_fname,'r') as mf:
        for i,troi in enumerate(troi_list):
            group = mf[g_path_tpl.format(troi)]
            tth_troi = tth_troi_list[i]
            chi_bins = np.linspace(tth_troi[0][0],tth_troi[0][1]+tth_troi[0][0],101)
            tth_bins = np.linspace(tth_troi[1][0],tth_troi[1][1]+tth_troi[1][0],101)

            tth_all = np.zeros(shape=100)
            chi_all = np.zeros(shape=100)
            d2_tth_all = np.zeros(shape=100)
            d2_chi_all = np.zeros(shape=100)
            peaks_all=[]
            
            for key in group.keys():
                print('on {}, {}'.format(troi,key))
                tth_fit = np.asarray(group[key]['tth_fit'])
                mapshape = tth_fit.shape[0:2]

                tth_peaks = tth_fit.reshape(mapshape[0]*mapshape[1],4,3)[:,:,1].flatten()
                tth_peaks = tth_peaks[np.where(tth_peaks>1)]
                tth_hist = np.histogram(tth_peaks,tth_bins)
                tth_all += tth_hist[0]
                plt.plot(tth_hist[1][:-1],tth_hist[0])
                plt.savefig(savepath+savename_tpl.format(troi,'tth',int(float(key)*100)), transparent=True)
                plt.clf()

                chi_fit = np.asarray(group[key]['chi_fit'])
                chi_peaks = chi_fit.reshape(mapshape[0]*mapshape[1],4,3)[:,:,1].flatten()
                chi_peaks = chi_peaks[np.where(chi_peaks>1)]
                chi_hist = np.histogram(chi_peaks,chi_bins)
                chi_all += chi_hist[0]
                plt.plot(chi_hist[1][:-1],chi_hist[0])
                plt.savefig(savepath+savename_tpl.format(troi,'chi',int(float(key)*100)), transparent=True)
                plt.clf()
                
                d2_fit = np.asarray(group[key]['2d_fit'])
                d2_tth_peaks = d2_fit.reshape(mapshape[0]*mapshape[1],4,6)[:,:,1].flatten()
                d2_tth_peaks = d2_tth_peaks[np.where(d2_peaks_tth>1)]
                d2_tth_hist = np.histogram(d2_tth_peaks,d2_tth_bins)
                d2_tth_all += d2_tth_hist[0]
                plt.plot(d2_tth_hist[1][:-1],d2_tth_hist[0])
                plt.savefig(savepath+savename_tpl.format(troi,'d2_tth',int(float(key)*100)), transparent=True)
                plt.clf()
                
                d2_chi_peaks = d2_fit.reshape(mapshape[0]*mapshape[1],4,6)[:,:,0].flatten()
                d2_chi_peaks = d2_chi_peaks[np.where(d2_chi_peaks>1)]
                d2_chi_hist = np.histogram(d2_chi_peaks,d2_chi_bins)
                d2_chi_all += d2_chi_hist[0]
                plt.plot(d2_chi_hist[1][:-1],d2_chi_hist[0])
                plt.savefig(savepath+savename_tpl.format(troi,'d2_chi',int(float(key)*100)), transparent=True)
                plt.clf()

                plt.plot(d2_chi_peaks, d2_tth_peaks, 'rx')
                plt.savefig(savepath+savename_tpl.format(troi,'d2',int(float(key)*100)), transparent=True)
                plt.clf()
                peaks_all += zip(chi_peaks, tth_peaks)

            plt.plot(tth_hist[1][:-1],tth_all[0])
            plt.savefig(savepath+savename_tpl.format(troi,'tth_all',(0)), transparent=True)
            plt.clf()
            
            plt.plot(chi_hist[1][:-1],chi_all[0])
            plt.savefig(savepath+savename_tpl.format(troi,'chi_all',(0)), transparent=True)
            plt.clf()      
                
            plt.plot(d2_tth_hist[1][:-1],d2_tth_all[0])
            plt.savefig(savepath+savename_tpl.format(troi,'d2_tth_all',(0)), transparent=True)
            plt.clf()
            plt.plot(d2_chi_hist[1][:-1],d2_chi_all[0])
            plt.savefig(savepath+savename_tpl.format(troi,'d2_chi_all',(0)), transparent=True)
            plt.clf()
            plt.plot(np.asarray(peaks_all)[:,0],np.asarray(peaks_all)[:,1])
            plt.savefig(savepath+savename_tpl.format(troi,'d2_all',(0)), transparent=True)
            plt.clf()

if __name__ == '__main__':

    
    
    main()
