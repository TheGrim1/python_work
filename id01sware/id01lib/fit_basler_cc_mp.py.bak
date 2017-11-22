from __future__ import print_function
from __future__ import absolute_import
#----------------------------------------------------------------------
# Description: 
#   functions for analysing limatake multiple image acquisitions intensity fluctuations
#   crosscorrelation class - basler cameras only
#   
# Author: Steven Leake <steven.leake@esrf.fr>
# Created at: Fri 09. Jun 23:20:30 CET 2017
# Computer: 
# System: 
#----------------------------------------------------------------------
#----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as pl
import h5py
import pylab as pl
import sys
import os
from multiprocessing import Process, Lock, Queue

from id01lib import hdf5_writer as h5w
from . import ImageRegistration as IR
from PyMca5 import EdfFile

pixel_size = np.array([55, 55])  # microns

########################
# PURPOSE
########################
# fit basler camera output to test beam stability using a cross correlation
# class
########################
# TODO
########################
# ADD ROI - DONE
# should be a dump images option in the beamviewer GUI

########################
# FUNCTIONS
########################




def gauss_func(x, a, x0, sigma):

    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def COM(arr):
    """calc COM"""
    tot = np.sum(arr)
    dims = arr.shape
    loc = np.array([])
    grid_dims = []
    for axis in dims:
        grid_dims.append(slice(0, axis))
    grid = np.ogrid[grid_dims]
    for axis in grid:
        loc = np.r_[loc, np.sum(arr*axis)/np.double(tot)]
    return loc

def get_roi_sum(array,roi = [[0,516],[0,516]]):
    return np.sum(np.sum(array[:, roi[1][0]:roi[1][1],roi[0][0]:roi[0][1]],axis=2),axis=1)

def load_rois(fn):
    f=open(fn,'r')
    rois = {}
    for line in f.readlines():
        if line[0] != '#':
            rois[line.split()[0]]=[int(x) for x in line.split()[1:6]]
    f.close()
    return rois
    
def analyse_limatake(h5fn, dict_rois, x = "/scan_0000/data/time_of_frame", y = "/scan_0000/data/images",group="/scan_0000/analysis/",id=''):
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    
    colours = ['b','g','r','c','m','y','k','w','b','g','r','c','m','y','k','w','b','g','r','c','m','y','k','w'] 
    
    x = h5w.get_dataset(h5fn,key = x)
    y = h5w.get_dataset(h5fn,key = y)
   
    pl.figure()
    for roi in dict_rois.keys():
        tmp_y = get_roi_sum(y, [[dict_rois[roi][1],\
                                    dict_rois[roi][2]],\
                                   [dict_rois[roi][3],\
                                    dict_rois[roi][4]]])
        tmp_y = tmp_y/np.mean(tmp_y)
        pl.plot(x, tmp_y, label = roi)
        h5w.add_dataset(h5fn,x,group,key = roi+"/x")
        #print(group+roi+"/x")
        h5w.add_dataset(h5fn,tmp_y,group,key = roi+"/y")    
        #print(group+roi+"/x")   
        
    pl.legend()
    pl.xlabel('Time(s)')
    pl.ylabel('Normalised Intensity')
    pl.savefig('roi_vs_time_'+id+'.pdf')
    pl.clf()
    
    pl.figure()
    for roi in dict_rois.keys():
        tmp_y = get_roi_sum(y, [[dict_rois[roi][1],\
                                    dict_rois[roi][2]],\
                                   [dict_rois[roi][3],\
                                    dict_rois[roi][4]]])
                                    
        N = tmp_y.shape[0]
        T = np.round(x[1]-x[0],2)
        yf = np.fft.fft(tmp_y)
        xf = np.linspace(0.0,1.0/(2.0*T),N/2)
        #print(xf[:],(2.0/N*np.abs(yf[:int(N/2)]))
        pl.plot(xf[:],2.0/N*np.abs(yf[:int(N/2)]),label = roi)
        h5w.add_dataset(h5fn,xf,group,key = roi+"/xf")
        h5w.add_dataset(h5fn,yf,group,key = roi+"/yf")  
    pl.xlim(0.1, x[-1])
    pl.ylim(0,np.max(2.0/N*np.abs(yf[2:int(N/2)])))
    pl.legend()
    pl.title('Beam Frequencies')
    pl.xlabel('Frequency (Hz)')
    pl.ylabel('Intensity (a.u)')
    pl.savefig('roi_freqs_'+id+'.pdf')
    pl.clf()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(np.sum(y[:,:,:],axis=0),norm=LogNorm())
    plt.colorbar(im)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    currentAxis = plt.gca()
    for i,roi in enumerate(dict_rois.keys()):
        currentAxis.add_patch(Rectangle((dict_rois[roi][1], dict_rois[roi][3]), dict_rois[roi][2]-dict_rois[roi][1], dict_rois[roi][4]-dict_rois[roi][3], fill= None, alpha=1,edgecolor=colours[i]))#, label='roi1')
        ax.annotate(roi, xy=(dict_rois[roi][1], dict_rois[roi][3]), xycoords='data',\
                    xytext=(-30,15), textcoords='offset points',\
                    arrowprops=dict(arrowstyle="->"))
    plt.savefig('sum_images_'+id+'.pdf')
    plt.clf()

########################
# CLASSES
########################


def CCworker(ref_data, tmp_data, time_stamp, time_zero, dump):
        """worker function"""
        # do the cross correlation
        x, y = IR.GetImageRegistration(ref_data, tmp_data, precision=1000)
        # print x,y, time_stamp-time_zero
        sys.stdout.write('.')
        # define your output
        dump.put([x, y, time_stamp-time_zero])
        return


class CrossCorrelator():

    def __init__(self, ref_im_fn, all_im_fns=[], h5fn="CC.h5", roi=[[0, 966], [0, 1296]]):
        self.ref_fn = ref_im_fn
        self.all_image_fns = all_im_fns  # list
        self.output = np.zeros(0, dtype=[('x', 'f8'), ('y', 'f8'), ('timestamp', 'f8')])
        self.output1 = np.zeros(0, dtype=[('timestamp', 'f8'), ('max', 'f8'), ('sum', 'f8'),
                                          ('com_x', 'f8'), ('com_y', 'f8'), ('peak_x', 'f8'),
                                          ('peak_y', 'f8'), ('mpx1x4', 'f8,f8,f8,f8')])
        self.h5fn = h5fn
        self.roi = roi

    def get_ref_im(self,image_number=0):
        pl.figure(1)
        self.data = EdfFile.EdfFile(ref_im_fn)
        _ref_im = self.data.GetData(image_number)[self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]]
        pl.imshow(_ref_im)
        pl.show()
        pl.clf()

    def set_ref_im(self, image_number=0):
        sys.stdout.write('opening file \n')
        self.data = EdfFile.EdfFile(ref_im_fn)
        sys.stdout.write('opening file complete\n')
        if self.ref_fn.endswith('.edf'):
            # single file so load it
            self.ref_im = self.data.GetData()[self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]]
            self.time_zero = np.double(self.data.GetHeader()['time_of_day'])
        elif self.ref_fn.endswith('.edf.gz'):
            # multiframe - pick the right frame
            self.ref_im = self.data.GetData(image_number)[self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]]
            self.time_zero = np.double(self.data.GetHeader(image_number)['time_of_day'])

    def multi_CC(self, fn='', lim_A=250, lim_B=10000, output=np.zeros(0, dtype=[('x', 'f8'),
                                                                                ('y', 'f8'),
                                                                                ('timestamp', 'f8')]), stats=False):
        """do CC on a edf.gz with multiple processors return the CC, as a structured array x,y,timestamp
        lim_A = should never exceed 250 on 16 cores w/ 32GB memory
        lim_b = defines the number of images you want to cross correlate
        """
        _dump = Queue()
        _tot_ims = output.shape[0]
        # print 'totims:',_tot_ims
        _l = Lock()
        _processes = []
        # if stats:
        # self.output1 = np.zeros(0,dtype = [('timestamp', 'f8'), ('max', 'f8'),
        # ('sum', 'f8'), ('com_x', 'f8'), ('com_y', 'f8'), ('peak_x', 'f8'), ('peak_y', 'f8')])

        ii = 0
        # if file already open use it don't repeat it
        if fn == self.ref_fn:
            _data = self.data
        else:
            _data = EdfFile.EdfFile(fn)

        # set the number of images to save
        if lim_B-_tot_ims >= _data.NumImages:
            _no_ims_fn = xrange(_data.NumImages)
            # print _no_ims_fn
        else:
            _no_ims_fn = xrange(lim_B-_tot_ims)
            # print _no_ims_fn

        _l.acquire()
        sys.stdout.write('%i images in file %s\n' % (_data.NumImages, fn))
        sys.stdout.write('%i:%i of %i\n' % (ii, _tot_ims+_data.NumImages, lim_B))
        _l.release()
        for ii in _no_ims_fn:
            # _l.acquire()
            # sys.stdout.write('%i,' % (ii))
            # _l.release()

            # get the data
            _im2CC = _data.GetData(ii)[self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]]
            _timestamp = np.double(_data.GetHeader(ii)['time_of_day'])

            # other stats
            if stats:
                stat = self.compress_xy_fitgauss_max_sum(_im2CC, _timestamp)
                self.output1 = np.r_[self.output1, np.array([(stat[0], stat[1], stat[2],
                                                              stat[3], stat[4], stat[5],
                                                              stat[6], (stat[7][0], stat[7][1],
                                                                        stat[7][2], stat[7][3]))],
                                                            dtype=[('timestamp', 'f8'), ('max', 'f8'), ('sum', 'f8'),
                                                                   ('com_x', 'f8'), ('com_y', 'f8'), ('peak_x', 'f8'),
                                                                   ('peak_y', 'f8'), ('mpx1x4', 'f8,f8,f8,f8')])]

            # send to multiple processors
            p = Process(target=CCworker, args=(self.ref_im, _im2CC, _timestamp, self.time_zero, _dump))
            p.start()
            _processes.append(p)

            # dump the data every time you hit the limit
            if ii % lim_A == lim_A-1 and ii != 0:
                sys.stdout.write('\n%i : %i\n' % (ii+_tot_ims, lim_B))
                for p in _processes:
                    p.join()

                # sys.stdout.write('Retrieving output\n')

                output = self.dump_queue(_dump, output=output)

        # if you have a non integer * limit number of points catch the leftovers and save them
        # if ii == _data.NumImages-1:
        #    sys.stdout.write('\n%i : %i\n' % (ii, lim_A))
        for p in _processes:
            p.join()

        # sys.stdout.write('\nRetrieving output\n')

        # iter = xrange(ii-ii % lim_A, ii)
        output = self.dump_queue(_dump, output=output)
        # update tot image no.
        _tot_ims += ii
        if stats:
            return output, self.output1
        else:
            return output

    def multi_fn_multi_CC(self, lim_A=250, lim_B=10000,
                          output=np.zeros(0, dtype=[('x', 'f8'),
                                                    ('y', 'f8'),
                                                    ('timestamp', 'f8')]), stats=False):
        if stats:
            for fn in self.all_image_fns:
                self.output, self.output1 = self.multi_CC(fn, lim_A, lim_B, output=self.output, stats=stats)
            return self.output, self.output1
        else:
            for fn in self.all_image_fns:
                self.output = self.multi_CC(fn, lim_A, lim_B, output=self.output, stats=stats)
            return self.output

    def dump_queue(self, queue, output=np.zeros(0, dtype=[('x', 'f8'),
                                                          ('y', 'f8'),
                                                          ('timestamp', 'f8')])):
        while True:
            # print queue.qsize()
            if int(queue.qsize()) == 0:
                break
            a = queue.get()
            output = np.r_[output, np.array([(a[0], a[1], a[2])],
                                            dtype=[('x', 'f8'), ('y', 'f8'), ('timestamp', 'f8')])]
        # print output
        return output

    def compress_xy_fitgauss_max_sum(self, tmp_data, _timestamp):
        x1 = np.linspace(0, 965, 966)
        data_x = tmp_data.sum(axis=1)
        # print x1.shape,data_x.shape
        x2 = np.linspace(0, 1295, 1296)
        data_y = tmp_data.sum(axis=0)
        # print x2.shape,data_y.shape

        com_x, com_y = COM(tmp_data)
        # fake data
        # data_x= gauss_func(x1,1,420,20)
        # data_y= gauss_func(x2,1,67,15)
        # data_y=np.where(data_y<0.05,0,data_y)
        # x2=np.where(data_y<0.05,0,x2)

        # Plot out the current state of the data and model
        # fig = pl.figure()
        # ax1 = fig.add_subplot(211)
        # ax1.plot(x1, data_x, c='b', label='integrated along y')
        # ax2 = fig.add_subplot(212)
        # ax2.plot(x2, data_y, c='b', label='integrated along x')
        # fig.savefig('model_and_noise.png')

        try:
            # Executing curve_fit
            popt1, pcov1 = curve_fit(gauss_func, x1, data_x)
            # y1 = gauss_func(x1, popt1[0], popt1[1], popt1[2])
            popt2, pcov2 = curve_fit(gauss_func, x2, data_y)
            # y2 = gauss_func(x2, popt2[0], popt2[1], popt2[2])

            # print popt # a,x0,sigma
            # ax1.plot(x1, y1, c='r', label='Best fit')
            # ax2.plot(x2, y2, c='r', label='Best fit')
            # ax1.legend()
            # ax2.legend()
            # fig.savefig('model_fit.png')
        except:
            # print fn, ' not fitted'
            popt1 = [0., 0., 0.]
            popt2 = [0., 0., 0.]

        # output
        return [_timestamp-self.time_zero, tmp_data.max(),
                tmp_data.sum(), com_x, com_y, popt1[1], popt2[1],
                [tmp_data[:483, :648].sum(), tmp_data[:483, 648:].sum(),
                 tmp_data[483:, :648].sum(), tmp_data[483:, 648:].sum()]]

    def dump_hdf5(self,):
        ID01_h5 = h5py.File(self.h5fn, 'w')
        if self.output.shape != 0:
            self.output.sort(order = 'timestamp')
            ID01_h5['cc_x'] = self.output['x']
            ID01_h5['cc_y'] = self.output['y']
            ID01_h5['timestamp'] = self.output['timestamp']
            ID01_h5['output_cc'] = self.output
        if self.output1.shape != 0:
            self.output1.sort(order = 'timestamp')
            ID01_h5['max'] = self.output1['max']
            ID01_h5['sum'] = self.output1['sum']
            ID01_h5['com_x'] = self.output1['com_x']
            ID01_h5['com_y'] = self.output1['com_y']
            ID01_h5['peak_x'] = self.output1['peak_x']
            ID01_h5['peak_y'] = self.output1['peak_y']
            ID01_h5['mpx1x4'] = self.output1['mpx1x4']
            ID01_h5['timestamp_stats'] = self.output1['timestamp']
            ID01_h5['output_stats'] = self.output1
        ID01_h5.close()

    def plot_pdf(self, fn="peak_pos_cc.pdf", pixel_size=np.array([20, 20])):
        pl.figure(1)
        pl.plot(self.output['timestamp'], self.output['x']*pixel_size[0], 'b', label="Peak_x")
        pl.plot(self.output['timestamp'], self.output['y']*pixel_size[1], 'r--', label="Peak_y")
        pl.xlabel("Time(s)")
        pl.ylabel("Position (um)")
        pl.legend()
        pl.savefig(fn)
        print(("peak(x) mean:", np.mean(self.output['x']*pixel_size[0]), "std: ", \
            np.std(self.output['x']*pixel_size[0])))
        print(("peak(y) mean:", np.mean(self.output['y']*pixel_size[1]), "std: ", \
            np.std(self.output['y']*pixel_size[1])))
        pl.clf()


def plot_max(h5fn, out_fn="max.pdf"):

    ID01_h5 = h5py.File(h5fn, 'r')
    pl.figure(1)
    x = ID01_h5['timestamp_stats']
    pl.plot(x[:ID01_h5['max'].shape[0]]-x[0], ID01_h5['max'][:]/np.mean(ID01_h5['max'][:]), 'b', label="max")
    pl.title(h5fn)
    pl.xlabel("Time(s)")
    pl.ylabel("Intensity (normalised)")
    pl.legend()
    pl.savefig(out_fn)
    pl.clf()


def plot_sum(h5fn, out_fn="sum.pdf"):

    ID01_h5 = h5py.File(h5fn, 'r')
    pl.figure(1)
    x = ID01_h5['timestamp_stats']
    pl.plot(x[:ID01_h5['sum'].shape[0]]-x[0], ID01_h5['sum'][:]/np.mean(ID01_h5['sum'][:]), 'b', label="sum")
    pl.title(h5fn)
    pl.xlabel("Time(s)")
    pl.ylabel("Intensity (normalised)")
    pl.legend()
    pl.savefig(out_fn)
    pl.clf()


def plot_com(h5fn, out_fn="com.pdf", pix_size=[20, 20]):

    ID01_h5 = h5py.File(h5fn, 'r')
    pl.figure(1)
    x = ID01_h5['timestamp_stats']
    pl.plot(x[:ID01_h5['com_x'].shape[0]]-x[0], (ID01_h5['com_x'][:]-ID01_h5['com_x'][0])*pix_size[0], 'b', label="com_x")
    pl.plot(x[:ID01_h5['com_y'].shape[0]]-x[0], (ID01_h5['com_y'][:]-ID01_h5['com_y'][0])*pix_size[1], 'r', label="com_y")
    pl.title(h5fn)
    pl.xlabel("Time(s)")
    pl.ylabel("Deviation (Microns)")
    pl.legend()
    pl.savefig(out_fn)
    pl.clf()


def plot_peak(h5fn, out_fn="peak.pdf", pix_size=[20, 20]):

    ID01_h5 = h5py.File(h5fn, 'r')
    pl.figure(1)
    x = ID01_h5['timestamp_stats']
    pl.plot(x[:ID01_h5['peak_x'].shape[0]]-x[0], (ID01_h5['peak_x'][:]-ID01_h5['peak_x'][0])*pix_size[0], 'b', label="peak_x")
    pl.plot(x[:ID01_h5['peak_y'].shape[0]]-x[0], (ID01_h5['peak_y'][:]-ID01_h5['peak_y'][0])*pix_size[1], 'r', label="peak_y")
    pl.title(h5fn)
    pl.xlabel("Time(s)")
    pl.ylabel("Deviation (Microns)")
    pl.legend()
    pl.savefig(out_fn)
    pl.clf()


def plot_mpxsum(h5fn, out_fn="mpxsum.pdf"):

    ID01_h5 = h5py.File(h5fn, 'r')
    pl.figure(1)
    x = ID01_h5['timestamp_stats']
    tmp = (ID01_h5['mpx1x4']['f0']+ID01_h5['mpx1x4']['f1'])-(ID01_h5['mpx1x4']['f2']+ID01_h5['mpx1x4']['f3'])  # left/right
    pl.plot(x[:tmp.shape[0]]-x[0], tmp[:]/np.mean(tmp), 'b', label="left-right")
    tmp = (ID01_h5['mpx1x4']['f0']+ID01_h5['mpx1x4']['f2'])-(ID01_h5['mpx1x4']['f1']+ID01_h5['mpx1x4']['f3'])  # top/bot
    pl.plot(x[:tmp.shape[0]]-x[0], tmp[:]/np.mean(tmp), 'r', label="top-bot")
    pl.title(h5fn)
    pl.xlabel("Time(s)")
    pl.ylabel("Intensity (normalised to mean)")
    pl.legend()
    pl.savefig(out_fn)
    pl.clf()

    # ffts
    # x = ID01_h5['timestamp']
    # x = x[:tmp.shape[0]]-x[0]
    # x = np.linspace(0.0,1.0/(2.0*(x[1]-x[0])),x.shape[0]/2)
    # tmp = (ID01_h5['mpx1x4']['f0']+ID01_h5['mpx1x4']['f1'])-(ID01_h5['mpx1x4']['f2']+ID01_h5['mpx1x4']['f3'])
    # y = np.fft.fft(tmp)
    # print x.shape, y.shape, np.abs(y)[:x.shape[0]/2].shape
    # pl.plot(x, np.abs(y)[:x.shape[0]], 'b', label="left-right")
    # tmp = (ID01_h5['mpx1x4']['f0']+ID01_h5['mpx1x4']['f2'])-(ID01_h5['mpx1x4']['f1']+ID01_h5['mpx1x4']['f3']) # top/bot
    # y = np.fft.fft(tmp)
    # pl.plot(x, np.abs(y)[:x.shape[0]], 'r', label="top-bot")
    # pl.xlim(0.1, x[-1])
    # pl.savefig(out_fn.split('.')[0]+'_fft.pdf')
    # pl.clf()

def plot_fft_quelquechose(h5fn, key='', fmt='.pdf', show=True):
    # fast for interactive file browsing
    ID01_h5 = h5py.File(h5fn, 'r')
    if key == '':
        print(ID01_h5.keys())
        key = raw_input('provide a key from the list above: ')

    pl.figure(1)
    # we have irregular data - beware really should be doing a non-uniform fft
    x = ID01_h5['timestamp_stats']
    x1 = np.linspace(0.0,1.0/(2.0*(x[1]-x[0])), x.shape[0]/2)
    y1 = np.abs(np.fft.fft(ID01_h5[key][:]))[:x.shape[0]/2]
    pl.plot(x1,  y1, 'r--', label=key)
    pl.title(h5fn)
    pl.xlabel("Freq (Hz)")
    pl.ylabel("Intensity")
    pl.legend()
    pl.xlim(0.1, x1[-1])
    pl.ylim(0,y1[5:].max())
    out_fn = h5fn.split('.')[0]+'_'+key+fmt
    print(out_fn)
    if show:
        pl.show()
    else:
        pl.savefig(out_fn)
    pl.clf()

def plot_quelquechose(h5fn, key='', fmt='.pdf', show=True):
    # fast for interactive file browsing
    ID01_h5 = h5py.File(h5fn, 'r')
    if key == '':
        print(ID01_h5.keys())
        key = raw_input('provide a key from the list above: ')

    pl.figure(1)
    x = ID01_h5['timestamp_stats']
    pl.plot(x[:ID01_h5[key].shape[0]]-x[0], ID01_h5[key][:]/np.mean(ID01_h5[key][:]), 'r--', label=key)
    pl.title(h5fn)
    pl.xlabel("Time(s)")
    pl.ylabel("Intensity")
    pl.legend()
    #pl.xlim(0.1, x[-1])
    out_fn = h5fn.split('.')[0]+'_'+key+fmt
    print(out_fn)
    if show:
        pl.show()
    else:
        pl.savefig(out_fn)
    pl.clf()

"""
########################
# variables
########################
# bveh pixel dims
pixel_size = np.array([20, 20])  # microns
#exposure = 0.7  # seconds - this just sets the x axis so corresponds to exposure + spec overheads

#specdir = '/mntdirect/_data_id01_inhouse/leake/projects/2015/commissioning/'
# cc stability
imagedir = '/mntdirect/_data_id01_inhouse/UPBLcomm/WBM_20150409/'
image_prefix = 'bveh-test_bveh'
image_suffix = "_%05d.edf.gz"
h5fn = 'stab_cc.h5'
ref_im_fn = imagedir+image_prefix+image_suffix % 17
fnos = [17, 18, 19, 20, 21]

# dcm stability
imagedir = '/mntdirect/_data_id01_inhouse/UPBLcomm/WBM_20150409/stab_dcm_bveh/'
image_prefix = 'stab_dcm_bveh'
image_suffix = "_%04d.edf.gz"
h5fn = 'stab_dcm.h5'
ref_im_fn = imagedir+image_prefix+image_suffix % 0
fnos = [0, 1, 2]

# combined dcm + WBM
imagedir = '/mntdirect/_data_id01_inhouse/UPBLcomm/WBM_20150409/stab_dcm_WBM_bveh/'
image_prefix = 'stab_dcm_WBM'
image_suffix = "_%04d.edf.gz"
h5fn = 'stab_dcm_WBM.h5'
ref_im_fn = imagedir+image_prefix+image_suffix % 1
fnos = [1, 2, 3, 4, 5, 6, 7]

files = os.listdir(imagedir)

all_im_fns = []


########################
# CODE
########################
for no in fnos:
    all_im_fns.append(imagedir+image_prefix+image_suffix % no)

cc = CrossCorrelator(ref_im_fn, all_im_fns, h5fn=h5fn)
cc.set_ref_im()
# cc.get_ref_im() # check the ROI you chose

# output = cc.multi_CC(fn=all_im_fns[0], lim_A=20, lim_B=45)
# output = cc.multi_CC(fn=all_im_fns[0], lima_A=20, lim_B=45,
#                      output=np.array([(1., 1., 1.)],
#                      dtype=[('x', 'f8'), ('y', 'f8'), ('timestamp', 'f8')]))

output, output1 = cc.multi_fn_multi_CC(lim_A=250, lim_B=10000, stats=True)
output.sort(order='timestamp')
print output.shape

cc.dump_hdf5()
cc.plot_pdf(fn="peak_pos_cc.pdf")

plot_max(h5fn, out_fn="max.pdf")
plot_sum(h5fn, out_fn="sum.pdf")
plot_com(h5fn, out_fn="com.pdf", pix_size=pixel_size)
plot_peak(h5fn, out_fn="peak.pdf", pix_size=pixel_size)
plot_mpxsum(h5fn, out_fn="mpxsum.pdf")
# plot_quelquechose(h5fn, key = '', fmt = '.pdf', show=True)
plot_fft_quelquechose(h5fn, key = '', fmt = '.pdf', show=True)


"""