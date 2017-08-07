#----------------------------------------------------------------------
# Description: 
#   analyse limatake multiple image acquisitions intensity fluctuations
#   
# Author: Steven Leake <steven.leake@esrf.fr>
# Created at: Fri 09. Jun 23:00:30 CET 2017
# Computer: 
# System: 
#----------------------------------------------------------------------
#----------------------------------------------------------------------

from id01lib import fit_basler_cc_mp as fit_cc
import numpy as np
import h5py
from id01lib import hdf5_writer as h5w
import pylab as pl

    
# Analyse intensity fluctuations
imagedir = '/mntdirect/_data_id01_inhouse/startup_20160126/images/aircon/'
image_prefix = 'freqdata1473431758_mpx4_'
image_suffix = "%05d.edf.gz"

h5fn = 'something.h5'
fnos = [26835]
#ref_im_fn = imagedir+image_prefix+image_suffix % fnos[0]

all_im_fns = []
for no in fnos:
    all_im_fns.append(imagedir+image_prefix+image_suffix % no)

h5w.edfsmf2hdf5(all_im_fns, out_fn = h5fn)
#h5file = h5py.File(h5fn,'r')
#data = get_dataset(h5fn,key = "/scan_0000/data/images")
#time_stamp = get_dataset(h5fn,key = "/scan_0000/data/time_of_frame")

dict_rois = fit_cc.load_rois('rois4aircon')
fit_cc.analyse_limatake(h5fn,dict_rois,id = 'something')

'''
#Analyse limatake basler
cc = fit_cc.CrossCorrelator(ref_im_fn, all_im_fns, h5fn=h5fn,roi=[[0, 516], [0, 516]])
cc.set_ref_im(ref_im_fn)
cc.get_ref_im() # check the ROI you chose

# output = cc.multi_CC(fn=all_im_fns[0], lim_A=20, lim_B=45)
# output = cc.multi_CC(fn=all_im_fns[0], lim_A=20, lim_B=45,
#                      output=np.array([(1., 1., 1.)],
#                      dtype=[('x', 'f8'), ('y', 'f8'), ('timestamp', 'f8')]))

output, output1 = cc.multi_fn_multi_CC(lim_A=50, lim_B=10000, stats=True)
output.sort(order='timestamp')
print(output.shape)

cc.dump_hdf5()
cc.plot_pdf(fn="peak_pos_cc.pdf")

crosscorrelator.plot_max(h5fn, out_fn="max.pdf")
crosscorrelator.plot_sum(h5fn, out_fn="sum.pdf")
crosscorrelator.plot_com(h5fn, out_fn="com.pdf", pix_size=pixel_size)
crosscorrelator.plot_peak(h5fn, out_fn="peak.pdf", pix_size=pixel_size)
crosscorrelator.plot_mpxsum(h5fn, out_fn="mpxsum.pdf")
# crosscorrelator.plot_quelquechose(h5fn, key = '', fmt = '.pdf', show=True)
crosscorrelator.plot_fft_quelquechose(h5fn, key = '', fmt = '.pdf', show=True)
'''

