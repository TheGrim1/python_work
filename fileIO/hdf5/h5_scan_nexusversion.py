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
from xrayutilities import lam2en as lam2en
import nexusformat.nexus as nx
from multiprocessing import Pool
import glob

# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy
from fileIO.spec.open_scan import spec_mesh
from fileIO.hdf5.open_h5 import open_h5
from fileIO.pyFAI.poni_for_troi import poni_for_troi
from fileIO.hdf5.h5_tools import get_shape
from fileIO.hdf5.h5_tools import filter_relevant_peaks
import fileIO.hdf5.nexus_update_functions as nuf
import fileIO.hdf5.nexus_tools as nxt
import pythonmisc.pickle_utils as pu
from pythonmisc.parallel_timer import parallel_timer
import fileIO.hdf5.copy_troi_worker as cptw
import fileIO.hdf5.sum_data_worker as sdw
import fileIO.hdf5.integrate_data_worker as idw
import fileIO.edf.write_edf_from_h5_single_file as h5toedf
import fileIO.edf.open_edf as open_edf


# from simplecalc.gauss_fitting import do_variable_gaussbkg_fit, do_variable_gauss_fit, do_multi_gauss_fit
# replaced with:
from simplecalc.gauss_fitting import do_variable_gaussbkg_pipeline

        


class h5_scan_nexus(object):
    '''
    Does some integration and collection of data so that eg. XSOCS can use this info 
    '''
    
# functions on self --------------------------------------
    
    def __init__(self,
                 data_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/DATA/AUTO-TRANSFER/eiger1/AJ2c_after_T2_yzth_1580_393_data_000000.h5',
                 calibration_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/aj_log/calib/calib1_prelim.poni',
                 save_directory = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/SESSION23/',
                 verbose=False):
        self.data_fname = data_fname
        self.master_fname = self.parse_master_fname()
        self.calibration_fname = calibration_fname
        self.save_dir = save_directory
        self.nx_f = self.make_h5()
        self.nx_f.close()
        print(self.save_dir)
        self.update_data(verbose=verbose)
        self.trois = {}

        
        self.noprocesses = 5
        '''
        self.data.update({'roi_real': np.zeros(shape=(0,0))})
        self.data.update({'roi_q'   : np.zeros(shape=(0,0))})
        self.data.update({'i_over_q': np.zeros(shape=(0,0))})
        self.data.update({'xrf'     : np.zeros(shape=(0,0))})
        self.data.update({'spec'    : {}})
        self.data.update({'meta'    : {'fnamelist'  :[fname]}})
        self.data.update({'shift'   : np.zeros(shape=(0,0))})      # TODO 2xframe array of shift acccording to [[/data/id13/inhouse2/AJ/skript/simplecalc/image_align.py]]
        self.data.update({'peaks_rad': np.zeros(shape=(0,0))})      # array of [a1, mu1, sigma1, a2, mu2 etc] x nframes}
        self.data.update({'peaks_azim': np.zeros(shape=(0,0))})     # TODO array of [a1, mu1, sigma1, a2, mu2 etc] x nframes}
        '''
    def get_eigerrunno(self):
        '''
        parsed from self.master_fname
        '''

        master_fname = self.master_fname
        eiger_runno = int(master_fname.split('_')[-2])
        return eiger_runno
        
    def parse_master_fname(self):
        data_fname = self.data_fname
        master_path = os.path.dirname(data_fname)
        master_fname = os.path.basename(data_fname)[:os.path.basename(data_fname).find("data")]+'master.h5'
        return master_path + os.path.sep + master_fname
        
    def update_data(self, verbose = False):
        nx_g = self.nx_f['entry/instrument']
        if verbose:
            print('reading ponifile at %s' %self.calibration_fname)
        nx_g = nuf.update_from_ponifile(nx_g,
                                        properties={'fname':self.calibration_fname})

        nx_g = nuf.update_from_eiger_master(nx_g,
                                            properties={'fname':self.master_fname},
                                            verbose=verbose)
        self.close_file()
        

    def get_data_shape(self, datagroup, troi, verbose=False):
        datashape = [0,0,0]
        datakey_list = datagroup.keys()
        if verbose:
            print('in get_data_shape, got datakey_list: ')
            print(datakey_list)
            
        for datakey in datakey_list:
            try:
                dataset = datagroup[datakey]
                if verbose:
                    print(type(dataset))
                    print(datakey + ' has shape:')
                    print(dataset.shape)
                    
                datashape[0]+=dataset.shape[0]
            except KeyError:
                print('Non-existant dataset: %s' % datakey)
                
                
        datatype = dataset.dtype
        datashape[1] = (troi[1][0])
        datashape[2] = (troi[1][1])
        return datashape, datatype
        
    def add_troi(self, troiname, troi):
        nx_troiprocess = nx.NXprocess(name=troiname)
        nx_troiprocess.insert(nx.NXfield(name ='troi', value = troi))
        self.integrated.insert(nx_troiprocess)
        self.trois.update({troiname:{}})
    
    def create_poni_for_trois(self, verbose =False):
        if verbose:
            print('reading ponifile:')
            print(self.nx_f['entry/instrument/calibration/Eiger4M'].attrs['ponifile_original_path'])
        orignal_poni_fname = self.nx_f['entry/instrument/calibration/Eiger4M'].attrs['ponifile_original_path']
        troipath = os.path.dirname(self.fname)
        
        for troiname in self.integrated.keys():
            troi = np.asarray(self.integrated[troiname]['troi'])
            (dummy, troiponifname) = poni_for_troi(orignal_poni_fname, troi, troiname, troipath)

            nx_g = self.integrated[troiname]
            nx_g = nuf.update_from_ponifile(nx_g,
                                            properties={'fname':troiponifname}, groupname=troiname)

        if verbose:
            print('creating ponifile:')
            print(troiponifname)

    def fit_azim_self(self, verbose = False, nopeaks =5):
        # TODO
        pass
    
    def fit_rad_self(self, plot = False, verbose = False, threshold = 0.05, nopeaks = 5, minwidth = 2, maxwidth = None):
        print('\nFITTING\n')
        '''
        TO BE TESTED
        '''


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

    
    def integrate_self(self, verbose = False):
        print('\nINTEGRATING, verbose = %s\n' % verbose)
        integ_starttime = time.time()
        total_datalength = 0
        noprocesses = self.noprocesses
        
        with h5py.File(self.fname) as h5_file:
            todolist = []
            for troiname in h5_file['entry/integrated'].keys():

                troi_grouppath = 'entry/integrated/{}'.format(troiname)
                h5troi_group = h5_file[troi_grouppath]
                troiponi_fname = h5troi_group['calibration'][troiname].attrs['ponifile_original_path']
                raw_datasetpath = 'entry/integrated/{}/raw_data/data'.format(troiname)
                troi = np.asarray(h5troi_group['troi'])
                
                datashape = h5_file[raw_datasetpath].shape 
                dtype = h5_file[raw_datasetpath].dtype
                datalength = datashape[0]

                
                npt_rad  = datashape[2]
                
                npt_azim = datashape[1]

                # setup groups and datasets
                q1D_group = self.nx_f[troi_grouppath]['q_integration_1D'] = nx.NXdata()
                tth1D_group = self.nx_f[troi_grouppath]['tth_integration_1D'] = nx.NXdata()
                q2D_group = self.nx_f[troi_grouppath]['q_integration_2D'] = nx.NXdata()
                tth2D_group = self.nx_f[troi_grouppath]['tth_integration_2D'] = nx.NXdata()

                self.nx_f.close()
                h5_file.flush()
                
                h5troi_group['q_integration_1D'].create_dataset(name = 'I_radial', dtype = dtype ,shape=(datashape[0],npt_rad))
                h5troi_group['tth_integration_1D'].create_dataset(name = 'I', dtype = dtype ,shape=(datashape[0],npt_rad))
                h5troi_group['q_integration_2D'].create_dataset(name = 'data', dtype = dtype ,shape=(datashape[0],npt_azim,npt_rad))
                h5troi_group['tth_integration_2D'].create_dataset(name = 'data', dtype = dtype, shape=(datashape[0],npt_azim,npt_rad))

                h5_file.flush()
                
                if 'mask' in h5troi_group.keys():
                    mask = h5troi_group['mask']
                else:
                    mask = np.zeros(shape = (datashape[1],datashape[2]))

                # do one integration to get the attributes out of ai:
                ai = pyFAI.AzimuthalIntegrator()              
                ai.load(troiponi_fname)

                frame = 0
                raw_data = h5troi_group['raw_data/data'][frame]

                q_data, qrange, qazimrange  = ai.integrate2d(data = raw_data, mask=mask, npt_azim = npt_azim ,npt_rad = npt_rad , unit='q_nm^-1')
                tth_data, tthrange, tthazimrange  = ai.integrate2d(data = raw_data, mask=mask, npt_azim = npt_azim ,npt_rad = npt_rad , unit='2th_deg')

                # wavelength = h5_troigroup['calibration'][troiname]['Wavelength'].value
                # energy =  lam2en(wavelength*1e-10)/1000
                qtroi      = xy_to_troi(min(qazimrange),max(qazimrange),min(qrange),max(qrange))
                tthtroi    = xy_to_troi(min(tthazimrange),max(tthazimrange),min(tthrange),max(tthrange))

                if verbose:
                    print('q-unit = {}, qtroi = '.format('q_nm^-1'))
                    print(qtroi)
                    print('qrange    = from %s to %s'% (max(qrange),min(qrange)))
                    print('azimrange = from %s to %s'% (max(qazimrange),min(qazimrange)))
                    print('2Theta range = from %s to %s' %(max(tthrange),min(tthrange)))
                    print('2Thetaazim range = from %s to %s' %(max(tthazimrange),min(tthazimrange)))

                h5troi_group['q_integration_1D'].create_dataset(name = 'qtroi', data = qtroi)
                h5troi_group['q_integration_1D'].create_dataset(name = 'frame_no', data = np.arange(datalength))
                h5troi_group['q_integration_1D'].create_dataset(name = 'q_radial', data = qrange)
                h5troi_group['q_integration_1D'].attrs['signal'] = 'I_radial'
                h5troi_group['q_integration_1D'].attrs['axes'] = ['frame_no','q_radial']
                h5troi_group['q_integration_1D/q_radial'].attrs['units'] = 'q_nm^-1'

                h5troi_group['tth_integration_1D'].create_dataset(name = 'tthtroi', data = tthtroi)
                h5troi_group['tth_integration_1D'].create_dataset(name = 'frame_no', data = np.arange(datalength))
                h5troi_group['tth_integration_1D'].create_dataset(name = 'tth', data = tthrange)
                h5troi_group['tth_integration_1D'].attrs['signal'] = 'I'
                h5troi_group['tth_integration_1D'].attrs['axes'] = ['frame_no','tth']
                h5troi_group['tth_integration_1D']['tth'].attrs['units'] = 'tth_deg'

                h5troi_group['q_integration_2D'].create_dataset(name='troi', data=qtroi)
                h5troi_group['q_integration_2D'].create_dataset(name='frame_no', data=np.asarray(range(datashape[0])))
                h5troi_group['q_integration_2D'].create_dataset(name='radial_range', data=qrange)
                h5troi_group['q_integration_2D'].create_dataset(name='azim_range', data=qazimrange)
                h5troi_group['q_integration_2D']['troi'].attrs['units'] = 'q_nm^-1'
                h5troi_group['q_integration_2D']['radial_range'].attrs['units'] = 'q_nm^-1'
                h5troi_group['q_integration_2D']['azim_range'].attrs['units']= 'deg'
                h5troi_group['q_integration_2D'].attrs['signal'] = 'data'
                h5troi_group['q_integration_2D'].attrs['axes'] = ['frame_no', 'qrange', 'azim_range']

                h5troi_group['tth_integration_2D'].create_dataset(name = 'troi', data = tthtroi)
                h5troi_group['tth_integration_2D'].create_dataset(name = 'frame_no', data = np.asarray(range(datashape[0])))
                h5troi_group['tth_integration_2D'].create_dataset(name = 'radial_range', data = tthrange)
                h5troi_group['tth_integration_2D'].create_dataset(name = 'azim_range', data = tthazimrange)
                h5troi_group['tth_integration_2D']['troi'].attrs['units']='2th_deg'
                h5troi_group['tth_integration_2D']['radial_range'].attrs['units']='2th_deg'
                h5troi_group['tth_integration_2D']['azim_range'].attrs['units']='deg'
                h5troi_group['tth_integration_2D'].attrs['signal'] = 'data'
                h5troi_group['tth_integration_2D'].attrs['axes'] = ['frame_no', 'qrange', 'azim_range']


                # now setup for integration of all frames in parallel:
                
                datalength = datashape[0]

                total_datalength += datalength
                target_fname = source_fname = self.fname
                
                # split up the indexes to be processed per worker:
                index_interval = int(datalength / noprocesses)
                source_index_list = [range(i*index_interval,(i+1)*index_interval) for i in range(noprocesses)]
                source_index_list[noprocesses-1] = range((noprocesses-1)*index_interval,datalength)
                target_index_list = source_index_list

                print('\nDEBUG\n')
                print('indexes for troi {}'.format(troiname))
                print('datalength')
                print(datalength)
                print('index_interval')
                print(index_interval)
                print('source_index_list')
                print(source_index_list)
                print('target_index_list')
                print(target_index_list)

                
                    
                
                # $PAR_TIME$ timer_fname_list  = [pt.newfile() for source_x in source_index_list]

                [todolist.append([target_fname,
                                  troi_grouppath,
                                  target_index_list[i],
                                  source_fname,
                                  raw_datasetpath,
                                  source_index_list[i],
                                  None,
                                  troiponi_fname,
                                  [npt_azim, npt_rad],
                                  verbose]) for i in range(len(target_index_list))]

            h5_file.flush()    
        instruction_list = []

        # print('\nDEBUG\n')
        # for i,todo in enumerate(todolist):
        #     print('\n' + '--'*25 + 'TODO{}:'.format(i))
        #     for todo_item in todo:
        #         print(todo_item)


        # Checkes the h5 file, this seems to fix some filesytem issues:
        print('touching h5 file: {}'.format(os.path.exists(self.fname)))
        self.nx_f.close()
                
        # h5_file is CLOSED... this seems to work:
        for i,todo in enumerate(todolist):
            instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
            instruction_list.append(instruction_fname)

        pool = Pool(processes=noprocesses)
        pool.map(idw.integrate_data_employer,instruction_list)
        pool.close()
        pool.join()
            
        # Checkes the h5 file, this seems to fix some filesytem issues:
        print('touching h5 file: {}'.format(os.path.exists(self.fname)))
        self.nx_f.close()
                       
        integ_endtime = time.time()
        integ_time = (integ_endtime - integ_starttime)
        print('='*25)
        print('\ntime taken for integration of {} frames = {}'.format(total_datalength, integ_time))
        print(' = {} Hz\n'.format(total_datalength/integ_time))
        print('='*25) 
        

        
# file IO ----------------------------------------
        
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

    def read_all_trois(self, troiname_list= 'all', threshold = None,  verbose = False, test = False):
        if test:
            print('testing:')

        print('\nREADING DATA, verbose = %s\n' % verbose)

        if troiname_list == 'all':
            troiname_list = self.integrated.keys()

        for troiname in troiname_list:
            data = self.read_troi(troiname, threshold=threshold, verbose=verbose, test=test)


    def read_troi(self, troiname, threshold = None,  verbose = False, test = False):
        troi = np.asarray(self.integrated[troiname]['troi'])
        total_datalength = 0
        
        noprocesses = self.noprocesses
        print('reading troi {}, number of processes used: {}'.format(troiname, noprocesses))


        read_starttime = time.time()
        source_fname = self.nx_f.attrs['file_name']
        target_fname = self.nx_f.attrs['file_name']
        
        nx_raw_data = self.integrated[troiname]['raw_data'] = nx.NXdata()
        nx_raw_data = self.integrated[troiname]['sum_data'] = nx.NXdata()
        
        todolist = []

        ## POS nexus doesn't want to follow my links (see data-analysis mails 17.01.2018), so I need to use h5py to read data
        # dubious to reopen an h5 that is opened with nxload...
        with h5py.File(self.nx_f.attrs['file_name']) as h5_dubious:

            source_grouppath = 'entry/instrument/Eiger4M/data'
            source_maskpath = 'entry/instrument/Eiger4M/mask'
            source_group = h5_dubious[source_grouppath]
            
            datakey_list = source_group.keys()
            datakey_list.sort()
            

            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].attrs['signal'] = 'data'
            h5_dubious['entry/integrated/{}/sum_data'.format(troiname)].attrs['signal'] = 'data'
            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].attrs['axes']  = ['frame_no','px_vert','px_horz']
            h5_dubious['entry/integrated/{}/sum_data'.format(troiname)].attrs['axes']  = 'frame_no'
            datashape, datatype = self.get_data_shape(source_group, troi=troi, verbose=verbose)

            target_index = 0
            target_datasetpath = 'entry/integrated/{}/raw_data/data'.format(troiname)

            if test:
                # TEST for tests read only 30 frames, errors if number of frames lower than number of processes!:
                test_no_frames = 30

                if verbose:
                    print('TEST: only copying {} frames.'.format(test_no_frames))
                    
                datakey = datakey_list[0]
                datashape = (test_no_frames, datashape[1], datashape[2])
                source_dataset = source_group['data_000001']
                datalength = test_no_frames
                total_datalength += datalength
                        
                source_datasetpath = source_grouppath + '/' +datakey
                     
                # split up the indexes to be processed per worker:
                index_interval = int(datalength / noprocesses)
                source_index_list = [range(i*index_interval,(i+1)*index_interval) for i in range(noprocesses)]
                source_index_list[noprocesses-1] = range((noprocesses-1)*index_interval,datalength)
                        
                target_index_list = [[x + target_index for x in source_x] for source_x in source_index_list]                 

                [todolist.append([target_fname,
                                  target_datasetpath,
                                  target_index_list[i],
                                  source_fname,
                                  source_datasetpath,
                                  source_index_list[i],
                                  source_maskpath,
                                  troi,
                                  verbose]) for i in range(len(target_index_list))]
                
                        
            # REAL (not TEST)
            else:

                if verbose:
                    print('reading ' + troiname + ' data %s' %datatype + ' of shape %s'%datashape)


                ### because I only look at the master file this can run into KeyErrors if the scan was aborted midway:
                try:

                    for datakey in datakey_list:

                        source_dataset = source_group[datakey]
                        datalength = source_dataset.shape[0]
                        total_datalength += datalength

                        source_datasetpath = source_grouppath + '/' +datakey
                     
                        # split up the indexes to be processed per worker:
                        index_interval = int(datalength / noprocesses)
                        source_index_list = [range(i*index_interval,(i+1)*index_interval) for i in range(noprocesses)]
                        source_index_list[noprocesses-1] = range((noprocesses-1)*index_interval,datalength)
                        
                        target_index_list = [[x + target_index for x in source_x] for source_x in source_index_list]                 

                        [todolist.append([target_fname,
                                          target_datasetpath,
                                          target_index_list[i],
                                          source_fname,
                                          source_datasetpath,
                                          source_index_list[i],
                                          source_maskpath,
                                          troi,
                                          verbose]) for i in range(len(target_index_list))]
                
                        
                        
                        target_index += datalength
                          

                except KeyError:
                    print('Non-existant dataset: %s' % datakey)
                    
            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].create_dataset(shape=datashape, dtype=datatype, name='data')
            h5_dubious['entry/integrated/{}/sum_data'.format(troiname)].create_dataset(shape=(datashape[0],), dtype=datatype, name='data')
            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].create_dataset(data=np.arange(datashape[0]), name='frame_no')
            h5_dubious['entry/integrated/{}/sum_data'.format(troiname)].create_dataset(data=np.arange(datashape[0]), name='frame_no')
            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].create_dataset(data=np.arange(datashape[1]), name='px_vert')
            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].create_dataset(data=np.arange(datashape[2]), name='px_horz')

            h5_dubious.flush()

        # Checkes the h5 file, this seems to fix some filesytem issues:
        print('touching h5 file: {}'.format(os.path.exists(self.fname)))


        instruction_list = []            
            
        for i,todo in enumerate(todolist):

            instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
            instruction_list.append(instruction_fname)

        pool = Pool(processes=noprocesses)
        pool.map(cptw.copy_troi_employer,instruction_list)
        pool.close()
        pool.join()

                
        # with h5py.File(self.fname) as h5_file:
        #     h5_file.flush()
            

        with h5py.File(self.nx_f.attrs['file_name']) as h5_dubious:
            h5_dubious.flush()

        # Checkes the h5 file, this seems to fix some filesytem issues:
            
        print('touching h5 file: {}'.format(os.path.exists(self.fname)))
        self.sum_troi_datset(troiname, verbose)
        
                
            
        read_endtime = time.time()
        read_time = (read_endtime - read_starttime)
        print('='*25)
        print('\ntime taken for reading and summation of {} frames = {}'.format(total_datalength, read_time))
        print(' = {} Hz\n'.format(total_datalength/read_time))
        print('='*25)


            
    def sum_all_trois(self, troiname_list= 'all', threshold = None,  verbose = False, test = False):
        if test:
            print('testing:')

        print('\nSUMMING DATA, verbose = %s\n' % verbose)

        if troiname_list == 'all':
            troiname_list = self.integrated.keys()

        for troiname in troiname_list:
            data = self.read_troi(troiname, threshold=threshold, verbose=verbose, test=test)

            
    def sum_troi_datset(self, troiname, verbose):


        noprocesses = self.noprocesses
        sum_fname = self.fname
        sum_datasetpath = 'entry/integrated/{}/sum_data/data'.format(troiname)

        with h5py.File(self.nx_f.attrs['file_name']) as h5_dubious:
            datashape = h5_dubious['entry/integrated/{}/raw_data/data'.format(troiname)].shape
            
            sum_interval = int(datashape[0] / noprocesses)
            sum_index_list = [range(i*sum_interval,(i+1)*sum_interval) for i in range(noprocesses)]
            sum_index_list[noprocesses-1] = range((noprocesses-1)*sum_interval,datashape[0])
            summation_list =[(sum_fname, sum_datasetpath, sum_indexes, verbose) for
                             sum_indexes in sum_index_list]
            h5_dubious.flush()


        # Checkes the h5 file. This seems to fix some filesytem issues:        
        print('the h5 file exists: {}'.format(os.path.exists(self.fname)))
        pool = Pool(processes=noprocesses)
        sum_data = pool.map(sdw.sum_data_worker, summation_list)
        
        with h5py.File(self.nx_f.attrs['file_name']) as h5_dubious:
            for results in sum_data:
                for frame, frame_sum in results:
                    h5_dubious[sum_datasetpath][frame]=frame_sum

            h5_dubious.flush()
        # Checkes the h5 file. This seems to fix some filesytem issues:        
        print('the h5 file exists: {}'.format(os.path.exists(self.fname)))

    def add_mask(self, mask_fname, verbose=False):
        '''
        add an associated mask file to this dataset. assumes .edf (for now)
        '''
        mask_data = open_edf.open_edf(mask_fname)
        with h5py.File(self.fname) as h5_file:
            h5_file['entry/instrument/Eiger4M'].create_dataset(name = 'mask', data = mask_data)
            h5_file.flush()
            
        # Checkes the h5 file, this seems to fix some filesytem issues:
        print('touching h5 file: {}'.format(os.path.exists(self.fname)))    
            
        if verbose:
            print('adding mask from file \n{}'.format(mask_fname))

        
        
    def make_edfs(self, groupname='tth_integration_2D', verbose=False):
        '''
        create xsocs-like edf file from an integrated dataset identified by groupname
        '''
        
        print('\nEXTRACTING EDFs, verbose = %s\n' % verbose)
        
        edf_starttime = time.time()
        total_datalength = 0
        noprocesses = self.noprocesses

        
        todolist = []
        
        
        with h5py.File(self.fname) as h5_file:
            for troiname in h5_file['entry/integrated'].keys():

        
                edf_fname = os.path.dirname(self.fname)+os.path.sep + troiname + '_{:05d}.edf.gz'.format(self.get_eigerrunno())
                print('edf fname = {}'.format(edf_fname))
        
                
                h5_datasetpath = 'entry/integrated/{}/{}/data'.format(troiname,groupname)
                h5_dataset = h5_file[h5_datasetpath]
                datalength = h5_dataset.shape[0]
                total_datalength += datalength
                

                target_index_list = source_index_list = [range(datalength)]
            
                [todolist.append([self.fname,
                                  h5_datasetpath,
                                  source_index_list[i],
                                  edf_fname,
                                  target_index_list[i],
                                  None,
                                  verbose]) for i in range(len(target_index_list))]

            h5_file.flush()
        # Checkes the h5 file. This seems to fix some filesytem issues:        
        print('the h5 file exists: {}'.format(os.path.exists(self.fname)))
            
        instruction_list = []
        
        for i,todo in enumerate(todolist):
            instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
            instruction_list.append(instruction_fname)

        pool = Pool(processes=noprocesses)
        pool.map(h5toedf.write_data_to_edf_employer, instruction_list)
        pool.close()
        pool.join()
        
        
        edf_endtime = time.time()
        edf_time = (edf_endtime - edf_starttime)
        print('='*25)
        print('\ntime taken for writing edfs of {} frames = {}'.format(total_datalength, edf_time))
        print(' = {} Hz\n'.format(total_datalength/edf_time))
        print('='*25) 

    def make_h5(self,
                fname = None, 
                default = False,
                verbose = False):
        '''
        write this class as <fname> or <dataprefix>_integrated.h5\n
        '''
        print('\ncreating file\n')

        if fname == None:
            fname    = self.data_fname
            if default:
                filename = os.path.basename(fname)[:os.path.basename(fname).find('data')]+'default.h5'        
            else:
                filename = os.path.basename(fname)[:os.path.basename(fname).find('data')]+'integrated.h5'

            datadir  = self.save_dir + os.path.sep + os.path.basename(fname)[0:os.path.basename(fname).find('_data')]
   
            if not os.path.exists(datadir):
                try:
                    os.mkdir(self.save_dir)
                except OSError:
                    os.mkdir(datadir)
                
            fname    = os.path.sep.join([datadir,filename])

        fname     = os.path.realpath(fname)
        savedir   = os.path.dirname(fname)
        
        if not os.path.exists(savedir):
            os.mkdir(savedir)
            print("making directory %s" % savedir)
    
        print('writing to file:\n%s'%fname)
        
        nx_f = nx.nxload(fname, 'w')
        nx_f.attrs['file_name']        = fname
        self.fname                     = fname
        nx_f.attrs['creator']          = os.path.basename(__file__)
        nx_f.attrs['HDF5_Version']     = h5py.version.hdf5_version
        nx_f.attrs['NX_class']         = 'NXroot'
        nxt.timestamp(nx_f)
        self.nxentry = nx_f['entry'] = nx.NXentry()
        nx_f['entry'].insert(nx.NXinstrument(name='instrument'))
        self.nxinstrument = nx_f['entry/instrument']
        nx_f['entry'].insert(nx.NXcollection(name='integrated'))
        self.integrated = nx_f['entry/integrated']

        
        return nx_f
        

    def close_file(self):
        '''
        self.nx_f.close()
        flushes file to disk (AFAIK)
        '''
        ## Does not seem to impact the access to the file in self.nx_f = ?

        self.nx_f.close()

        return 

def do_r1_w3():
    
    find_tpl = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/DATA/AUTO-TRANSFER/eiger1/r1_w3_xzth__**_data_**4**'
    calib_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/calib/calib3.poni'
    all_datafiles = glob.glob(find_tpl)
    troi_tr = ((72, 1609), (260, 294))
    troi_ml = ((1196, 80), (415, 526))
    save_dir = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_xzth/'
    mask_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/testmask.edf'
    print('*'*25 + '\n'*5 + '*'*25)
    
    for i, data_fname in enumerate(all_datafiles):
        doer = h5_scan_nexus(data_fname = data_fname,
                             calibration_fname = calib_fname,
                             save_directory = save_dir,
                             verbose=True)

        doer.add_troi('troi_tr',((72, 1609), (260, 294)))
        doer.add_troi('troi_ml',((1196, 80), (415, 526)))
        test = False
        doer.noprocesses = 4
        verbose = False
        doer.add_mask(mask_fname, verbose=verbose)
        doer.create_poni_for_trois(verbose= verbose)
        doer.read_all_trois(verbose=verbose, test=test)
        doer.integrate_self(verbose=verbose)
        doer.make_edfs(groupname='tth_integration_2D', verbose=verbose)
        doer.close_file()
 
        
if __name__ == "__main__":

    do_r1_w3()
    
    # doer = h5_scan_nexus(data_fname = '/data/id13/inhouse2/AJ/skript/xsocs/my_example/r1_w3_E63/h5/r1_w3_test_63_data_000001.h5',
    #                      calibration_fname = '/data/id13/inhouse2/AJ/skript/xsocs/my_example/r1_w3_E63/calib1.poni',
    #                      save_directory = '/data/id13/inhouse2/AJ/skript/xsocs/my_example/r1_w3_E63/integrated',
    #                      verbose=True)
    # # test = h5_scan_nexus(data_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/DATA/AUTO-TRANSFER/eiger1/r1_w3_test_63_data_000001.h5',
    # #                      calibration_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/calib/calib1.poni',
    # #                      verbose=True)      
    # doer.add_troi('troi1',((263, 800), (128, 118)))
    # test = False
    # doer.noprocesses=8
    # verbose = False
    # doer.create_poni_for_trois(verbose= verbose)
    # doer.read_all_trois(verbose=verbose, test = test)
    # doer.integrate_self(verbose=verbose)
    # doer.make_edfs('troi1','tth_integration_2D')
    # doer.close_file()
