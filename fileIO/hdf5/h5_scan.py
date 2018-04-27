from __future__ import print_function
# global imports
import h5py
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import json
import ast
import time
import pyFAI
from xrayutils import lam2en as lam2en
import nexusformat.nexus as nx

# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy
from fileIO.spec.open_scan import spec_mesh
from fileIO.hdf5.open_h5 import open_h5
from fileIO.pyFAI.poni_for_troi import poni_for_troi
from fileIO.hdf5.h5_tools import get_shape
from fileIO.hdf5.h5_tools import filter_relevant_peaks

#from simplecalc.gauss_fitting import do_variable_gaussbkg_fit, do_variable_gauss_fit, do_multi_gauss_fit
# replaced with:
from simplecalc.gauss_fitting import do_variable_gaussbkg_pipeline

class h5_scan(object):
    '''
    **** 'roi_real' - roi of peak in real space
    **** 'roi_q'    - roi of peak in q (1/nm)
    **** 'i_over_q' - azimuthal interation of the roi
    **** 'spec'     - dict of spec_mesh  [[/data/id13/inhouse2/AJ/skript/fileIO/spec/open_scan.py]]3
    **** 'meta' - dict with items:
        {'troi'}         : relative to Eiger
        {'troiname'}     :'troi1'
        {'poni'}         : path of poni file
        {'poni_for_troi'}: path of poni file for troi
        {'qroi'}         : similar to troi but in ['meta']['qunits']  (it is rectangular in q, not realspace)
        {'fnamelist'     : [paths to hdf5 datafiles]} maybe use update_files for this
        {'shift'}        : tuple of shift acccording to [[/data/id13/inhouse2/AJ/skript/simplecalc/image_align.py]]
        '''

# functions on self --------------------------------------
    
    def __init__(self, fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/DATA/AUTO-TRANSFER/eiger1/AJ2c_after_T2_yzth_1580_393_data_000000.h5'):
        self.data = {}
        self.data.update({'roi_real': np.zeros(shape=(0,0))})
        self.data.update({'roi_q'   : np.zeros(shape=(0,0))})
        self.data.update({'i_over_q': np.zeros(shape=(0,0))})
        self.data.update({'xrf'     : np.zeros(shape=(0,0))})
        self.data.update({'spec'    : {}})
        self.data.update({'meta'    : {'fnamelist'  :[fname]}})
        self.data.update({'shift'   : np.zeros(shape=(0,0))})      # TODO 2xframe array of shift acccording to [[/data/id13/inhouse2/AJ/skript/simplecalc/image_align.py]]
        self.data.update({'peaks_rad': np.zeros(shape=(0,0))})      # array of [a1, mu1, sigma1, a2, mu2 etc] x nframes}
        self.data.update({'peaks_azim': np.zeros(shape=(0,0))})     # TODO array of [a1, mu1, sigma1, a2, mu2 etc] x nframes}
        
    def update_meta(self, key, value):
        '''
        {'troi'}         : relative to Eiger
        {'troiname'}     :'troi1'
        {'poni'}         : path of poni file
        {'poni_for_troi'}: path of poni file for troi
        {'qroi'}         : similar to troi but in 1/nm space
        {'fnamelist'     : [paths to hdf5 datafiles]} maybe use update_files for this
        {'qunits'}       : default = 'q_nm^-1'
        {'scanlist'}     : global list of yzth scan (eg.) of which this scan is one internal number , eigernumber, specnumber
        {'path'}         : path where this class instance is saved as .h5
        '''
        self.data['meta'].update({key:value})

    def meta(self, key):
        '''
        return the value in data['meta']['key']
        '''
        return self.data['meta'][key]

    def data(self):
        '''
        return the value in data['key']
        '''
        return self.data['spec']

    def spec(self,key):
        '''
        shortcut to specfile info
        '''
        return self.data('spec')['key']
        
        
    def create_poni_for_troi(self,troi = None, troiname = None):
        if not troi     == None:
            self.update_meta('troi',troi)
        if not troiname == None:
            self.update_meta('troiname',troiname)
            
        troi     = self.data['meta']['troi']
        troiname = self.data['meta']['troiname']
        
        (dummy, troiponifname) = poni_for_troi(self.data['meta']['poni'],troi,troiname)
        self.update_meta('poni_for_troi',troiponifname)
        

    def fit_azim_self(self, verbose = False, nopeaks =5):
        # TODO
        pass
    
    def fit_rad_self(self, plot = False, verbose = False, threshold = 0.05, nopeaks = 5, minwidth = 2, maxwidth = None):
        print('\nFITTING\n')


        data     = np.copy(self.data['i_over_q'])
        times    = np.zeros(data.shape[2])
        residual = np.zeros(data.shape[2])
        peaks    = np.zeros(shape = (nopeaks,3,data.shape[2]))

        
        # for i in [480]: # for DEBUG
        for i in range(data.shape[2]):
        
            if verbose:
                print('fitting frame %s' % i)

            # timing:
            before = time.time()
            if max(data[:,:,i].flat)<threshold:
                if verbose:
                    print('No peak over threshold')
                pass
            else:
                foundpeaks, residual[i] = do_variable_gaussbkg_pipeline(data[:,:,i],
                                                                        nopeaks=nopeaks,
                                                                        plot=plot,
                                                                        verbose=verbose,
                                                                        threshold=threshold,
                                                                        minwidth=minwidth,
                                                                        maxwidth=maxwidth)
                print('foundpeaks.shape:')
                print(foundpeaks.shape)
                peaks[:,:,i] = filter_relevant_peaks(data[:,:,i],foundpeaks,verbose = verbose)
            timetaken = time.time()-before
            times[i]=timetaken
            if verbose:
                print('took %ss'%timetaken)
        self.data['peaks_rad']    = peaks
        self.data['residual_rad'] = residual
        return times

    def create_2d_plot_of_fitted_peaks(self):
        ##HERE TODO
        peakpos = self.data['peaks_rad']
        
        

    
    def integrate_self(self, verbose = False, npt_rad = None, npt_azim = None):
        print('\nINTEGRATING, verbose = %s\n' % verbose)

        self.create_poni_for_troi()
        
        ai = pyFAI.AzimuthalIntegrator()
        
        ai.load(self.data['meta']['poni_for_troi'])

        datashape = self.data['roi_real'].shape
        if npt_rad  == None:
            npt_rad  = datashape[1]
        if npt_azim == None:
            npt_azim = datashape[0]
            
        i_over_q = np.zeros(shape=[2,npt_rad,datashape[2]])
        roi_q    = np.zeros(shape=[npt_azim,npt_rad,datashape[2]])
        
        mask = np.zeros(shape = (datashape[0],datashape[1]))
        for frame in range(datashape[2]):
            if verbose:
                print('integrating frame: %s' % frame)
            data = self.data['roi_real'][:,:,frame]


            i_over_q[0,:,frame], i_over_q[1,:,frame] = ai.integrate1d(data = data, mask = mask, npt = npt_rad , unit=unit)

            roi_q[:,:,frame],qrange,azimqrange  = ai.integrate2d(data = data, mask = mask, npt_azim = npt_azim ,npt_rad = npt_rad , unit='q_nm^-1')

            roi_2th[:,:,frame],2thrange,azim2thrange  = ai.integrate2d(data = data, mask = mask, npt_azim = npt_azim ,npt_rad = npt_rad , unit='2th_deg')

        wavelength = ai.get_wavelength()
        energy =  lam2en(wavelength*1e-10)/1000
        qtroi                 = xy_to_troi(min(azimrange),max(azimrange),min(qrange),max(qrange))

        if verbose:
            print('q-unit = %s, qtroi = '%'q_nm^-1')
            print(qtroi)
            print('qrange    = from %s to %s'% (max(qrange),min(qrange)))
            print('azimrange = from %s to %s'% (max(azimrange),min(azimrange)))
            print('2Theta range = from %s to %s' %(max(2thrange),min(2thrange)))

        self.update_meta('qtroi',qtroi)
        self.update_meta('qunit','q_nm^-1')
        self.data.insert(nx.NXfield(name = 'I_radial', value = i_over_q[0:,:]))
        self.data.insert(nx.NXfield(name = 'q_radial', value = i_over_q[1:,:],units='q_nm^-1'))

        self.data.insert(nx.NXfield(name = 'frame_no', np.asarray(range(datashape[2]))))
        
        self.data.insert(nx.NXfield(name = 'roi_q', value = roi_q))
        self.data.insert(nx.NXfield(name = 'qrange', value = qrange,units='q_nm^-1'))
        self.data.insert(nx.NXfield(name = 'azimqrange', value = azimqrange,units='q_nm^-1'))
        
        
        self.data.insert(nx.NXfield(name = 'roi_2th', value = roi_2th))
        self.data.insert(nx.NXfield(name = '2thrange', value = qrange,units='2th_deg'))
        self.data.insert(nx.NXfield(name = 'azim2thrange', value = qrange,units='2th_deg'))

# file IO ----------------------------------------

    def update_files(self,verbose = False):

        fnamelist = []
        onefile   = self.meta('fnamelist')[0]
        nametpl   = onefile[:onefile.find('.h5')-3]+'%03d.h5'
        print(nametpl %1 )
        print(nametpl %1 )
        i = 1
        while os.path.exists(nametpl % i):
            fnamelist.append(os.path.realpath(nametpl % i))
            i+=1
        self.update_meta('fnamelist',fnamelist)
        if verbose:
            print('found files:')
            print(fnamelist)
        return (fnamelist)
        
    def update_spec(self, specfname = None,
                 scanno = None,
                 counter =  None):
        # HERE
        if specfname ==  None:
            specfname   = self.data['spec']['specfile']
        if scanno    == None:
            scanno      = self.data['spec']['scanno']
        if counter   == None:
            counter     = self.data['spec']['counter']
        
        self.data['spec'].update({'specfile'       :specfname})
        self.data['spec'].update({'scanno'         :scanno})
        self.data['spec'].update({'counter'        :counter})
        
        spec = spec_mesh(specfname, scanno, counter)
        spec.load()
        self.data['xrf'] = spec.data

        self.update_meta('shape'     ,spec.data.shape)
        self.update_meta('realshape' ,spec.info['realshape'])
        self.update_meta('Theta'     ,float(spec.info['Theta']))
        
        return spec.data

    def read_data(self, framesperfile = 2000, threshold = 65000,  verbose = False, test = False):
        if test:
            print('testing:')

        print('\nREADING DATA, verbose = %s\n' % verbose)
        
        self.update_files()
        troi      = self.data['meta']['troi']
        fnamelist = self.data['meta']['fnamelist']

        datashape = get_shape(fnamelist = fnamelist, troi = troi)
        if verbose:
            print('reading data of shape %s'%datashape)

        if test:
        # TEST for tests read only 1 frame:
            data      = np.zeros(shape = (datashape[0],datashape[1],1))
            for fileframe in range(1):
                fname = fnamelist[0]
                frame           = fileframe
                if verbose:
                    print('reading file %s' % fname)
                    print('reading frame %s'% frame)
                data[:,:,frame] = open_h5(fname, framelist = [fileframe], troi = troi, threshold = threshold, verbose = False)
        
        # REAL (not TEST)
        else:
            data      = np.zeros(shape = datashape) 
            for i, fname in enumerate(fnamelist):
                for fileframe in range(min((datashape[2] - i * framesperfile), framesperfile)):
                    
                    frame           = fileframe + framesperfile * i
                    if verbose:
                        print('reading file %s' % fname)
                        print('reading frame %s'% frame)
                    data[:,:,frame] = open_h5(fname, framelist = [fileframe], troi = troi, threshold = threshold)
                
                
        self.data['roi_real'] = data
                
    def write_self(self,
                   fname = None, 
                   default = False,
                   verbose = False):
        '''
        write this class as <fname> or <dataprefix>_integrated.h5\n
        dictionaries are saved as json.dumps strings and can be read by json.loads
        '''
        print('\nWRITING\n')

        if fname == None:
            fname    = self.data['meta']['fnamelist'][0]
            if default:
                filename = os.path.basename(fname)[:os.path.basename(fname).find('data')]+'default.h5'        
            else:
                filename = os.path.basename(fname)[:os.path.basename(fname).find('data')]+'integrated.h5'
            datadir  = fname[:fname.find('/DATA')] + '/PROCESS/SESSION23/integrated/' + fname[fname.find('eiger')+7:fname.find('eiger')+14]

            if not os.path.exists(datadir):
                try:
                    os.mkdir(fname[:fname.find('/DATA')]+'/PROCESS/SESSION23/')
                except OSError as msg:
                    print(msg)
                    print('continuing')
                try:
                    os.mkdir(fname[:fname.find('/DATA')]+'/PROCESS/SESSION23/integrated/')
                except OSError as msg:
                    print(msg)
                try:
                    os.mkdir(datadir)
                except OSError as msg:
                    print(msg)

            fname    = os.path.sep.join([datadir,filename])

        fname     = os.path.realpath(fname)
        savedir   = os.path.dirname(fname)
        
        if not os.path.exists(savedir):
            os.mkdir(savedir)
            print("making directory %s" % savedir)
            
        self.update_meta('path',fname)
        print('writing to file:\n%s'%fname)
        
        # if os.path.exists(fname):
        #     os.remove(fname)
        savefile         = h5py.File(fname,"w")
        group            = savefile.create_group('entry/data')
        for key in list(self.data.keys()):
            if type(self.data[key]) == dict or type(self.data[key]) == str:
                dt         = h5py.special_dtype(vlen=str)
                dataset    = json.dumps(self.data[key])
                group.create_dataset(key,data = dataset,dtype = dt)
            else:
                dataset    = self.data[key]
                group.create_dataset(key,data = dataset, compression = 'gzip', shuffle = True)

        savefile.flush()
        savefile.close()
        return fname

    def read_self(self, fname):
        '''
        read the <fname> file into this h5_mess class
        '''
        print('reading '+ fname)
        if fname.find(".h5") != -1:
            fname    = os.path.realpath(fname)
            readfile = h5py.File(fname, "r")
            readdata = readfile['entry/data/']
            for key in list(readdata.keys()):
                if readdata[key].dtype == 'O':
                    self.data.update({key: ast.literal_eval((readdata[key].value))})
                else:
                    self.data.update({key: readdata[key].value})
            readfile.flush()
            readfile.close()
                                
                    
        else:
            print( 'ERROR, this doesnt seem to be a h5 file: \n%s'% fname)
            
    def do_all(self,verbose = False):
        self.update_files(verbose = verbose)
        self.read_data(verbose = verbose)
        self.integrate_self(verbose = verbose)
        self.fit_rad_self(verbose = verbose)
        self.write_self(verbose = verbose)
    
