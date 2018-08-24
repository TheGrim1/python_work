import h5py
import sys
sys.path.append('/data/id13/inhouse2/AJ/skript/')
import fileIO.hdf5.h5_tools as h5t
from simplecalc.slicing import troi_to_slice            
import numpy as np

class master_getter(object):
    '''
    An iterator that returns frames from a scan identified by one master or data file name. 
    Only uses the links in the master file. Will return values untill no further frames are found. 
    Does not have len and shape.
    Can be started in a "for frame in master_getter(fname)" loop before all frames are recorded, after first datafile is complete. Will find frames recorded after init.
    accepts arguement troi ((topleft, topright), (height, width)) to read only this region
    '''
    def __init__(self, fname, verbose=False, troi=None):
        if fname.find('master.h5')>0:
            self.master_fname = fname
        elif fname.find('_data_')>0:
            self.master_fname = h5t.parse_master_fname(fname)
        self.verbose=verbose
        if self.verbose:
            print('opening {}'.format(self.master_fname))
        self.h5_f = h5py.File(self.master_fname,'r')
        self.data_group = self.h5_f['entry/data']

        self.read_troi=False
        if type(troi)!= type(None):
            self.troi_to_read = troi
            self.read_troi = True
            self.slice_to_read = troi_to_slice(troi)
            
        self.data_keys = self.data_group.keys()

        self.data_keys.sort()
        self.frames_per_file = self.data_group[self.data_keys[0]].shape[0]
        self.index = 0

    def __iter__(self):
        return self

    def next(self):
        self.index +=1
        item = self.index
        try:
            result = self.__getitem__(item)
            if type(result)==type(None):
                raise StopIteration()
            else:
                return result
        except KeyError:
            raise StopIteration()

    def __getitem__(self, item):
        dataset_no = int(item/self.frames_per_file)
        dataset_frame = item%self.frames_per_file
        data_key=self.data_keys[dataset_no]
        if self.verbose:
            print('getting frame {} from dataset {}'.format(item,data_key))
        if self.read_troi:
            try:
                return self.data_group[data_key][dataset_frame][self.slice_to_read]
            except TypeError:
                return None
        else:
            return self.data_group[data_key][dataset_frame]

        return result

    def close(self):
        if self.verbose:
            print('closing {}'.format(self.master_fname))
        self.h5_f.close()

    def open(self):
        if self.verbose:
            print('opening {}'.format(self.master_fname))
        self.h5_f = h5py.File(self.master_fname,'r') 
        
    def __del__(self):
        if self.verbose:
            print('closing {}'.format(self.master_fname))
        self.h5_f.close()

    def __exit__(self,a,b,c):
        self.__del__()

    def __enter__(self):
        return self
    
        
class data_getter(object):
    '''
    An iterator that returns frames from a scan identified by one master or data file name. 
    On init looks for datafiles that are there, returns frames via the links in the master file. 
    Has len and shape.    
    accepts arguement troi ((topleft, topright), (height, width)) to read only this region
    '''
    def __init__(self, fname, verbose=False, troi=None):
        if fname.find('master.h5')>0:
            self.master_fname = fname
        elif fname.find('_data_')>0:
            self.master_fname = h5t.parse_master_fname(fname)
        self.verbose=verbose
        if self.verbose:
            print('opening {}'.format(self.master_fname))
        self.h5_f = h5py.File(self.master_fname,'r')
        self.data_group = self.h5_f['entry/data']

        self.data_keys = self.data_group.keys()
        self.data_keys.sort()
        self.datasets = []

        self.read_troi=False
        if type(troi)!= type(None):
            self.troi_to_read = troi
            self.read_troi = True
            self.slice_to_read = troi_to_slice(troi)
        
        self.no_frames=0
        data_fname_tpl=h5t.parse_data_fname_tpl(self.master_fname)
        for i, data_key in enumerate(self.data_keys):
            data_fname = data_fname_tpl.format(i+1)
            try:
                with h5py.File(data_fname,'r') as data_h5:
                    self.no_frames += data_h5['entry/data/data'].shape[0]
                self.datasets.append(self.data_group[data_key])
            except IOError:
                print('Did not find expected datafile {}\nreturning smaller dataset'.format(data_fname))
        self.frames_per_file = self.datasets[0].shape[0]
        self.frames_in_last_file = (self.no_frames-1)%self.frames_per_file + 1

    def __getitem__(self, item):
        if type(item)==slice:

            if type(item.step)==type(None):
                step = 1
            else:
                step = item.step
            if type(item.stop)==type(None):
                stop = self.no_frames
            else:
                stop  = item.stop
            if type(item.start)==type(None):
                start = 0
            else:
                start = item.start
                
            start_dataset_no = int(start/self.frames_per_file)
            last_dataset_no  = int((stop-step)/self.frames_per_file)

            ds_start = start%self.frames_per_file
            ds_stop  = (stop -1)%self.frames_per_file + 1 


            if start_dataset_no == last_dataset_no:
                if self.verbose:
                    print('getting frames [{}:{}:{}] from dataset number {}'.format(ds_start, ds_stop, step, start_dataset_no))
                if self.read_troi:
                    return self.datasets[start_dataset_no][ds_start:ds_stop:step][self.slice_to_read]
                else:
                    return self.datasets[start_dataset_no][ds_start:ds_stop:step]
            elif (last_dataset_no-start_dataset_no)==1:
                remainder = ds_start%step
                if self.verbose:
                    print('getting frames [{}:{}:{}] from datasets {} and {}'.format(ds_start, ds_stop, step,start_dataset_no,stop_dataset_no))
                if self.read_troi:
                    return np.vstack([self.datasets[start_dataset_no][ds_start::step][self.slice_to_read],
                                      self.datasets[last_dataset_no][remainder:ds_stop:step][self.slice_to_read]])
                else:
                    return np.vstack([self.datasets[start_dataset_no][ds_start::step],
                                      self.datasets[last_dataset_no][remainder:ds_stop:step]])
              
            else:
                raise NotImplementedError('TODO')
            
                

    
        if item>=self.no_frames:
            raise IndexError('index: {} out of range for no_frames: {}'.format(item, self.no_frames))
        else:
            dataset_no = int(item/self.frames_per_file)
            dataset_frame = item%self.frames_per_file
            if self.verbose:
                print('getting frame {} from dataset number {}'.format(item,dataset_no))
            if self.read_troi:
                return self.datasets[dataset_no][dataset_frame][self.slice_to_read]
            else:
                return self.datasets[dataset_no][dataset_frame]


    def __len__(self):
        return self.no_frames

    @property
    def shape(self):
        return self.get_shape()
    
    def __del__(self):
        self.close()

    def __exit__(self,a,b,c):
        self.__del__()

    def __enter__(self):
        return self

    def close(self):
        if self.verbose:
            print('closing {}'.format(self.master_fname))
        self.h5_f.close()

    def open(self):
        if self.verbose:
            print('opening {}'.format(self.master_fname))
        self.h5_f = h5py.File(self.master_fname,'r')

    def get_shape(self):
        if self.read_troi:
            return tuple([self.no_frames]+list(self.troi_to_read[1]))
        else:
            return tuple([self.no_frames]+list(self.datasets[0].shape[1:]))
    



def main():
    test_fname = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-06-03_user_ls2785_truppault/DATA/AUTO-TRANSFER/eiger1/aj_test_247_data_000001.h5'
    
    dg = data_getter(test_fname, verbose = True,troi=[[0,0],[3,3]])
    print(dg.get_shape())
    for i in range(len(dg)):
         print(i,(dg[i]).sum())
    del(dg)
    mg = master_getter(test_fname, verbose=True,troi=[[0,0],[3,3]])
    for i, frame in enumerate(mg):
        print(i,(frame.shape))
    for i in range(3900,4100):
        print(i,(mg[i]).sum())
    del(mg)

    
if __name__=='__main__':
    main()        

    
