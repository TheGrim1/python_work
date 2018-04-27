
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
#from xrayutilities import lam2en as lam2en
import nexusformat.nexus as nx
from multiprocessing import Pool
import glob

# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.slicing import troi_to_slice, xy_to_troi, troi_to_xy
from fileIO.spec.open_scan import spec_mesh
from fileIO.hdf5.open_h5 import open_h5
from fileIO.pyFAI.poni_for_troi import poni_for_troi
from fileIO.hdf5.h5_tools import get_shape, get_datagroup_shape
from fileIO.hdf5.h5_tools import filter_relevant_peaks
from fileIO.hdf5.h5_tools import get_eigerrunno, parse_master_fname
import fileIO.hdf5.nexus_update_functions as nuf
import fileIO.hdf5.nexus_tools as nxt
import pythonmisc.pickle_utils as pu
from pythonmisc.parallel_timer import parallel_timer
import fileIO.hdf5.workers.copy_troi_worker as cptw
import fileIO.hdf5.workers.integrate_data_worker as idw
import fileIO.edf.write_edf_from_h5_single_file as h5toedf
import fileIO.edf.open_edf as open_edf


# from simplecalc.gauss_fitting import do_variable_gaussbkg_fit, do_variable_gauss_fit, do_multi_gauss_fit
# replaced with:
from simplecalc.gauss_fitting import do_variable_gaussbkg_pipeline

class h5_scan_nexus(object):
    '''
    Does some integration and collection of data so that eg. XSOCS can use this info 
    not XSOCS compatible ATM
    '''
    
# functions on self --------------------------------------
    
    def __init__(self):
        pass

    def setup_ID13(self,
                   data_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/DATA/AUTO-TRANSFER/eiger1/AJ2c_after_T2_yzth_1580_393_data_000000.h5',
                   spec_fname = None,
                   spec_scanno = 0,                   
                   calibration_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/aj_log/calib/calib1_prelim.poni',
                   save_directory = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/SESSION23/',
                   verbose=False):
        self.beamline = 'ID13'
        self.detector = 'Eiger4M'
        self.data_fname = data_fname
        self.master_fname = parse_master_fname(data_fname)
        self.calibration_fname = calibration_fname
        self.save_dir = save_directory
        self.nx_f = self.make_h5()
        self.nx_f.close()
        print('saving this work in:')
        print(self.save_dir)
        self.update_spec(spec_fname=spec_fname, scanno=spec_scanno, verbose=verbose)
        self.update_data(verbose=verbose)
        self.trois = {}
        self.noprocesses = 5


    def setup_ID01(self,
                   data_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/DATA/AUTO-TRANSFER/eiger1/AJ2c_after_T2_yzth_1580_393_data_000000.h5',
                   calibration_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/aj_log/calib/calib1_prelim.poni',
                   save_directory = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/PROCESS/SESSION23/',
                   verbose=False):

        self.beamline = 'ID01'
        self.detector = 'maxipix'
        self.data_fname = data_fname
        self.master_fname = parse_master_fname(data_fname)
        self.calibration_fname = calibration_fname
        self.save_dir = save_directory
        self.nx_f = self.make_h5()
        self.nx_f.close()
        print(self.save_dir)
        print('saving this work in:')
        self.update_data(verbose=verbose)
        self.trois = {}
        self.noprocesses = 5
        
        
    def update_data(self, verbose = False):
        nx_g = self.nx_f['entry/instrument']
        if verbose:
            print('reading ponifile at %s' %self.calibration_fname)
        nx_g = nuf.update_from_ponifile(nx_g,
                                        properties={'fname':self.calibration_fname})

        if self.beamline=='ID13':

            nx_g = nuf.update_from_eiger_master(nx_g,
                                                properties={'fname':self.master_fname},
                                                verbose=verbose)
        elif self.beamline=='ID01':
            nx_g = nuf.update_from_id01_master(nx_g,
                                               properties={'fname':self.master_fname},
                                               verbose=verbose)
        else:
            raise ValueError('unkown beamline {}'.format(self.beamline))
    
        self.close_file()
        
        
    def add_troi(self, troiname, troi):
        nx_troiprocess = nx.NXprocess(name=troiname)
        nx_troiprocess.insert(nx.NXfield(name ='troi', value = troi))
        self.integrated.insert(nx_troiprocess)
        self.trois.update({troiname:{}})
    
    def create_poni_for_trois(self, verbose =False):
        orignal_poni_fname = self.nx_f['entry/instrument/calibration/'+self.detector].attrs['ponifile_original_path']
        if verbose:
            print('reading ponifile:')
            

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

    
    def integrate_self(self, verbose = False):
       
        integ_starttime = time.time()
        total_datalength = 0
        noprocesses = self.noprocesses

        print('\nINTEGRATING, verbose = %s\n' % verbose)
        print('number of processes used: {}'.format(noprocesses))
        
        with h5py.File(self.fname) as h5_file:
            todolist = []
            for troiname in h5_file['entry/integrated'].keys():

                troi_grouppath = 'entry/integrated/{}'.format(troiname)
                h5troi_group = h5_file[troi_grouppath]
                troiponi_fname = h5troi_group['calibration'][troiname].attrs['ponifile_original_path']
                raw_datasetpath = 'entry/integrated/{}/raw_data/data'.format(troiname)
                troi = np.asarray(h5troi_group['troi'])
                
                datashape = h5_file[raw_datasetpath].shape 
                # dtype = h5_file[raw_datasetpath].dtype
                dtype = np.float32
                datalength = datashape[0]

                
                npt_rad  = int(datashape[2]*1.5)
                
                npt_azim = int(datashape[1]*1.5)

                # setup groups and datasets
                qrad_group = self.nx_f[troi_grouppath]['q_radial'] = nx.NXdata()
                chi_group = self.nx_f[troi_grouppath]['chi_azimuthal'] = nx.NXdata()
                tthrad_group = self.nx_f[troi_grouppath]['tth_radial'] = nx.NXdata()
                q2D_group = self.nx_f[troi_grouppath]['q_2D'] = nx.NXdata()
                tth2D_group = self.nx_f[troi_grouppath]['tth_2D'] = nx.NXdata()
                axes_group = self.nx_f[troi_grouppath]['axes'] = nx.NXcollection()
                
                self.nx_f.close()
                h5_file.flush()
                
                h5troi_group['q_radial'].create_dataset(name = 'I', dtype = dtype, shape=(datashape[0],npt_rad), compression='lzf', shuffle=True)
                h5troi_group['chi_azimuthal'].create_dataset(name = 'I', dtype = dtype, shape=(datashape[0],npt_azim), compression='lzf', shuffle=True)
                h5troi_group['tth_radial'].create_dataset(name = 'I', dtype = dtype, shape=(datashape[0],npt_rad), compression='lzf', shuffle=True)
                h5troi_group['q_2D'].create_dataset(name = 'data', dtype = dtype, shape=(datashape[0],npt_azim,npt_rad), compression='lzf', shuffle=True)
                h5troi_group['tth_2D'].create_dataset(name = 'data', dtype = dtype, shape=(datashape[0],npt_azim,npt_rad), compression='lzf', shuffle=True)


                h5_file.flush()
                
                if 'mask' in h5troi_group.keys():
                    mask = h5troi_group['mask']
                else:
                    mask = np.zeros(shape = (datashape[1],datashape[2]))

                # do one integration to get the attributes out of ai:
                ai = pyFAI.AzimuthalIntegrator()              
                ai.load(troiponi_fname)
                ai.setChiDiscAtZero()

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

                h5troi_group['axes'].create_dataset(name = 'frame_no', data=np.arange(datashape[0]))
                h5troi_group['axes'].create_dataset(name = 'q_radial', data=qrange)
                h5troi_group['axes/q_radial'].attrs['units'] = 'q_nm^-1'
                h5troi_group['axes'].create_dataset(name = 'tth', data=tthrange)
                h5troi_group['axes/tth'].attrs['units'] = 'tth_deg'
                h5troi_group['axes'].create_dataset(name = 'chi_azim', data=qazimrange)
                h5troi_group['axes/chi_azim'].attrs['units'] = 'deg'
                h5troi_group['axes'].create_dataset(name = 'troi', data=troi)
                h5troi_group['axes'].create_dataset(name = 'qtroi', data=qtroi)
                h5troi_group['axes'].create_dataset(name = 'tthtroi', data=tthtroi)
                    
                h5troi_group['q_radial/frame_no'] = h5troi_group['axes/frame_no']
                h5troi_group['q_radial/q_radial'] = h5troi_group['axes/q_radial']
                h5troi_group['q_radial'].attrs['signal'] = 'I'
                h5troi_group['q_radial'].attrs['axes'] = ['frame_no','q_radial']
                h5troi_group['q_radial/q_radial'].attrs['units'] = 'q_nm^-1'

                h5troi_group['chi_azimuthal/frame_no'] = h5troi_group['axes/frame_no']
                h5troi_group['chi_azimuthal/azim_range'] = h5troi_group['axes/chi_azim']
                h5troi_group['chi_azimuthal'].attrs['signal'] = 'I'
                h5troi_group['chi_azimuthal'].attrs['axes'] = ['frame_no','azim_range']

                h5troi_group['tth_radial/frame_no'] = h5troi_group['axes/frame_no']
                h5troi_group['tth_radial/tth'] = h5troi_group['axes/tth']
                h5troi_group['tth_radial'].attrs['signal'] = 'I'
                h5troi_group['tth_radial'].attrs['axes'] = ['frame_no','tth']

                h5troi_group['q_2D/frame_no'] = h5troi_group['axes/frame_no']
                h5troi_group['q_2D/q_radial'] = h5troi_group['axes/q_radial']
                h5troi_group['q_2D/azim_range'] = h5troi_group['axes/chi_azim']
                h5troi_group['q_2D'].attrs['signal'] = 'data'
                h5troi_group['q_2D'].attrs['axes'] = ['frame_no', 'azim_range', 'q_radial']

                h5troi_group['tth_2D/frame_no'] = h5troi_group['axes/frame_no']
                h5troi_group['tth_2D/tth'] = h5troi_group['axes/tth']
                h5troi_group['tth_2D/azim_range'] = h5troi_group['axes/chi_azim']
                h5troi_group['tth_2D'].attrs['signal'] = 'data'
                h5troi_group['tth_2D'].attrs['axes'] = ['frame_no', 'azim_range', 'tth']

                # now setup for integration of all frames in parallel:
                
                datalength = datashape[0]

                total_datalength += datalength
                target_fname = source_fname = self.fname
                
                # split up the indexes to be processed per worker:
                index_interval = int(datalength / noprocesses)
                source_index_list = [range(i*index_interval,(i+1)*index_interval) for i in range(noprocesses)]
                source_index_list[noprocesses-1] = range((noprocesses-1)*index_interval,datalength)
                target_index_list = source_index_list            
                    
    
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
        #     for todo_tem in todo:
        #         print(todo_item)


        # Checkes the h5 file, this seems to fix some filesytem issues:
        touch = os.path.exists(self.fname)
        # print('touching h5 file: {}'.format(touch))
        self.nx_f.close()
                
        # h5_file is CLOSED... this seems to work:
        for i,todo in enumerate(todolist):
            instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
            instruction_list.append(instruction_fname)

        if noprocesses==1:
            # DEBUG (change to employer for max performance
            for instruction in instruction_list:
                idw.integrate_data_employer(instruction)
            ## non parrallel version for one dataset and timing:
            #idw.integrate_data_worker(instruction_list[0])
        else:
            pool = Pool(processes=noprocesses)
            pool.map_async(idw.integrate_data_employer,instruction_list)
            pool.close()
            pool.join()

    
        # Checkes the h5 file, this seems to fix some filesytem issues:
        touch = os.path.exists(self.fname)
        # print('touching h5 file: {}'.format(touch))
        self.nx_f.close()
                       
        integ_endtime = time.time()
        integ_time = (integ_endtime - integ_starttime)
        print('='*25)
        print('\ntime taken for integration of {} frames = {}'.format(total_datalength, integ_time))
        print(' = {} Hz\n'.format(total_datalength/integ_time))
        print('='*25) 
        

        
# file IO ----------------------------------------
        
    def update_spec(self, spec_fname = None,
                    scanno = None,
                    counter_list =  None, verbose=False):
        
        nx_g = self.nx_f['entry/instrument']
        
        nx_g = nuf.update_final_spec(nx_g,
                                     properties={'fname':spec_fname,
                                                 'scanno':scanno},verbose=verbose)
        if self.beamline == 'ID13':
            self.do_ID13_scanparse(verbose=verbose)
        elif self.beamline == 'ID01':
            self.do_ID01_scanparse(verbose=verbose)

    def do_ID13_scanparse(self,verbose=False):
        pass
        # with h5py.File(self.fname) as h5_file:
        #     h5_file['entry']
                    
    def do_ID01_scanparse(self,verbose=False):
        raise NotImplementedError('ID01 specparse: TODO')
    
                    

        
            
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
        
        todolist = []

        ## POS nexus doesn't want to follow my links (see data-analysis mails 17.01.2018), so I need to use h5py to read data
        # dubious to reopen an h5 that is opened with nxload...
        with h5py.File(self.nx_f.attrs['file_name']) as h5_dubious:

            source_grouppath = 'entry/instrument/{}/data'.format(self.detector)
            source_maskpath = 'entry/instrument/{}/mask'.format(self.detector)
            source_group = h5_dubious[source_grouppath]
            
            datakey_list = source_group.keys()
            datakey_list.sort()
            

            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].attrs['signal'] = 'data'
            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].attrs['axes']  = ['frame_no','px_vert','px_horz']
            datashape, datatype = get_datagroup_shape(source_group, troi=troi, verbose=verbose)

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

                        source_datasetpath = source_grouppath + '/' + datakey
                     
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
                    
            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].create_dataset(shape=datashape, dtype=datatype, name='data', compression='lzf', shuffle=True)
            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].create_dataset(data=np.arange(datashape[0]), name='frame_no')
            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].create_dataset(data=np.arange(datashape[1]), name='px_vert')
            h5_dubious['entry/integrated/{}/raw_data'.format(troiname)].create_dataset(data=np.arange(datashape[2]), name='px_horz')

            h5_dubious.flush()

        # Checkes the h5 file, this seems to fix some filesytem issues:
        touch=os.path.exists(self.fname)
        # print('touching h5 file: {}'.format(touch))


        instruction_list = []            
            
        for i,todo in enumerate(todolist):

            instruction_fname = pu.pickle_to_file(todo, caller_id=os.getpid(), verbose=verbose, counter=i)
            instruction_list.append(instruction_fname)

       
        if noprocesses==1:
            # DEBUG (change to employer for max performance
            for instruction in instruction_list:
                cptw.copy_troi_employer(instruction)
            ## non parrallel version for one dataset and timing:
            #        ## debug timing:
            # cptw.copy_troi_worker(instruction_list[0])

        else:
            pool = Pool(processes=noprocesses)
            pool.map_async(cptw.copy_troi_employer,instruction_list)
            pool.close()
            pool.join()          

        # Checkes the h5 file, this seems to fix some filesytem issues:

        touch = os.path.exists(self.fname)
        # print('touching h5 file: {}'.format(touch))        
                
            
        read_endtime = time.time()
        read_time = (read_endtime - read_starttime)
        print('='*25)
        print('\ntime taken for reading of {} frames = {}'.format(total_datalength, read_time))
        print(' = {} Hz\n'.format(total_datalength/read_time))
        print('='*25)


    def add_mask(self, mask_fname, verbose=False):
        '''
        add an associated mask file to this dataset. assumes .edf (for now)
        '''
        mask_data = open_edf.open_edf(mask_fname)
        with h5py.File(self.fname) as h5_file:
            h5_file['entry/instrument/{}'.format(self.detector)].create_dataset(name = 'mask', data = mask_data, compression='lzf', shuffle=True)
            h5_file.flush()
            
        # Checkes the h5 file, this seems to fix some filesytem issues:
        touch = os.path.exists(self.fname)
        # print('touching h5 file: {}'.format(touch))    
            
        if verbose:
            print('adding mask from file \n{}'.format(mask_fname))

        
            

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
        nx_f.attrs['beamline']         = self.beamline
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


def do_read_rois(args):
    no_processes = args[0]
    find_tpl = args[1]
    spec_fname = args[2]
    spec_to_eiger_scanno_offset = args[3]
    save_dir = args[4]
    troi_dict = args[5]
    mask_fname = args[6]
    calib_fname = args[7]
    test = args[8]
    verbose = args[9]
                 
    all_datafiles = glob.glob(find_tpl)
    eigerscanno_list = [get_eigerrunno(parse_master_fname(x)) for x in all_datafiles]
    specscanno_list = [x + spec_to_eiger_scanno_offset for x in eigerscanno_list]

    
    todo_list = []

    
    for i, data_fname in enumerate(all_datafiles):
        todo = []
        todo.append(data_fname)
        todo.append(spec_fname)
        todo.append(specscanno_list[i])
        todo.append(calib_fname)
        todo.append(save_dir)
        todo.append(troi_dict)
        todo.append(mask_fname)
        todo.append(test)
        todo.append(verbose)
        
        todo_list.append(todo)
    
    pool = Pool(processes=no_processes)
    pool.map_async(do_read_one_h5, todo_list)
    pool.close()
    pool.join()

def do_read_one_h5(args):

    data_fname = args[0]
    spec_fname = args[1]
    spec_scanno = args[2]
    calib_fname = args[3]
    save_dir = args[4]
    troi_dict = args[5]
    mask_fname = args[6]
    test = args[7]
    verbose = args[8]

    print('process {} working on {}'.format(os.getpid(),data_fname))
    
    one_h5 = h5_scan_nexus()
    one_h5.setup_ID13(data_fname,
                      spec_fname,
                      spec_scanno=spec_scanno,
                      calibration_fname = calib_fname,
                      save_directory = save_dir,
                      verbose=True)

    for troiname,troi in troi_dict.items():
        one_h5.add_troi(troiname,troi)
        
    one_h5.noprocesses = 1
    one_h5.add_mask(mask_fname, verbose=verbose)
    one_h5.create_poni_for_trois(verbose= verbose)
    one_h5.read_all_trois(verbose=verbose, test=test)
    one_h5.integrate_self(verbose=verbose)
    #one_h5.make_edfs(groupname='tth_integration_2D', verbose=verbose)
    one_h5.close_file()
    return True    

def do_r1_w3_gpu2():
    find_tpl = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/DATA/AUTO-TRANSFER/eiger1/r1_w3_xzth__**_data_**4**'
    all_datafiles = glob.glob(find_tpl)

    spec_fname= '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/DATA/r1_w3/r1_w3.dat'
    spec_to_eiger_scanno_offset = 53-5
    eigerscanno_list = [get_eigerrunno(parse_master_fname(x)) for x in all_datafiles]
    specscanno_list = [x + spec_to_eiger_scanno_offset for x in eigerscanno_list]

    save_dir = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_gpu2/'

    troi_tr = ((72, 1609), (260, 294))
    troi_ml = ((1196, 80), (415, 526))
    troi_dict=dict([['troi_tr',troi_tr],['troi_ml',troi_ml]])
    
    mask_fname = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/testmask.edf'
    calib_fname = '/hz/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/calib/calib3.poni'
    test = False
    verbose = False
    
    print('*'*25 + '\n'*5 + '*'*25)

    # better to go parrallel at this point and not later:
    super_processes = 11
    print('starting {} processes to read and integrate data from'.format(super_processes))
    print(find_tpl)

    todo_list =[]
    
    for i, data_fname in enumerate(all_datafiles):
        todo = []
        todo.append(data_fname)
        todo.append(spec_fname)
        todo.append(specscanno_list[i])
        todo.append(calib_fname)
        todo.append(save_dir)
        todo.append(troi_dict)
        todo.append(mask_fname)
        todo.append(test)
        todo.append(verbose)
        
        todo_list.append(todo)

    # do_read_one_h5(todo_list[2])
    pool = Pool(processes=super_processes)
    pool.map_async(do_read_one_h5, todo_list)
    pool.close()
    pool.join()


    
def do_r1_w3():
    
    find_tpl = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/DATA/AUTO-TRANSFER/eiger1/r1_w3_xzth__**_data_**4**'
    all_datafiles = glob.glob(find_tpl)

    spec_fname= '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/DATA/r1_w3/r1_w3.dat'
    spec_to_eiger_scanno_offset = 53-5
    eigerscanno_list = [get_eigerrunno(parse_master_fname(x)) for x in all_datafiles]
    specscanno_list = [x + spec_to_eiger_scanno_offset for x in eigerscanno_list]

    save_dir = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/integrated/r1_w3_xzth/'

    troi_tr = ((72, 1609), (260, 294))
    troi_ml = ((1196, 80), (415, 526))
    troi_dict=dict([['troi_tr',troi_tr],['troi_ml',troi_ml]])
    
    mask_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/testmask.edf'
    calib_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/calib/calib3.poni'
    test = False
    verbose = False
    
    print('*'*25 + '\n'*5 + '*'*25)

    # better to go parrallel at this point and not later:
    super_processes = 11
    print('starting {} processes to read and integrate data from'.format(super_processes))
    print(find_tpl)

    todo_list =[]
    
    for i, data_fname in enumerate(all_datafiles):
        todo = []
        todo.append(data_fname)
        todo.append(spec_fname)
        todo.append(specscanno_list[i])
        todo.append(calib_fname)
        todo.append(save_dir)
        todo.append(troi_dict)
        todo.append(mask_fname)
        todo.append(test)
        todo.append(verbose)
        
        todo_list.append(todo)

    
    #do_read_one_h5(todo_list[0])
    pool = Pool(processes=super_processes)
    pool.map_async(do_read_one_h5, todo_list)
    pool.close()
    pool.join()
    

        
def do_calib_example():
    
    data_fname = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-08-21_inh_blc10829/DATA/AUTO-TRANSFER/eiger1/spheru_calib_al2o3_m60_40_data_000001.h5'
    calib_fname = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-08-21_inh_blc10829/PROCESS/SESSION25/calib_m60/a75_40_spheru_calib_al2o3_m60_max.poni'

    troi_tr = ((550,1750),(200,150))
    troi_ml = ((1000,80),(400,500))
    save_dir = '/data/id13/inhouse8/THEDATA_I8_1/d_2017-08-21_inh_blc10829/PROCESS/aj_log/integrated/calib_60m/'
    mask_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/testmask.edf'
    print('*'*25 + '\n'*5 + '*'*25)

    one_h5 = h5_scan_nexus()
    one_h5.setup_ID13(data_fname,
                      spec_fname='/data/id13/inhouse8/THEDATA_I8_1/d_2017-09-28_inh_ihmi1340/DATA/setup/setup.dat',
                      spec_scanno=1,
                      calibration_fname = calib_fname,
                      save_directory = save_dir,
                      verbose=True)

    
    one_h5.add_troi('troi_tr',troi_tr)
    one_h5.add_troi('troi_ml',troi_ml)
    test = False
    one_h5.noprocesses = 1
    verbose = False
    one_h5.add_mask(mask_fname, verbose=verbose)
    one_h5.create_poni_for_trois(verbose= verbose)
    one_h5.read_all_trois(verbose=verbose, test=test)
    one_h5.integrate_self(verbose=verbose)
    #one_h5.make_edfs(groupname='tth_integration_2D', verbose=verbose)
    one_h5.close_file()
    
if __name__ == "__main__":

    do_r1_w3_gpu2()
    # do_calib_example()
    # one_h5 = h5_scan_nexus(data_fname = '/data/id13/inhouse2/AJ/skript/xsocs/my_example/r1_w3_E63/h5/r1_w3_test_63_data_000001.h5',
    #                      calibration_fname = '/data/id13/inhouse2/AJ/skript/xsocs/my_example/r1_w3_E63/calib1.poni',
    #                      save_directory = '/data/id13/inhouse2/AJ/skript/xsocs/my_example/r1_w3_E63/integrated',
    #                      verbose=True)
    # # test = h5_scan_nexus(data_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/DATA/AUTO-TRANSFER/eiger1/r1_w3_test_63_data_000001.h5',
    # #                      calibration_fname = '/data/id13/inhouse6/THEDATA_I6_1/d_2016-10-27_in_hc2997/PROCESS/aj_log/calib/calib1.poni',
    # #                      verbose=True)      
    # one_h5.add_troi('troi1',((263, 800), (128, 118)))
    # test = False
    # one_h5.noprocesses=8
    # verbose = False
    # one_h5.create_poni_for_trois(verbose= verbose)
    # one_h5.read_all_trois(verbose=verbose, test = test)
    # one_h5.integrate_self(verbose=verbose)
    # one_h5.make_edfs('troi1','tth_integration_2D')
    # one_h5.close_file()
