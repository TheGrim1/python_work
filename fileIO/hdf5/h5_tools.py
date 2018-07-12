from __future__ import print_function
import h5py
import numpy as np
import sys, os
from subprocess import check_output

# local imports
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from simplecalc.calc import add_peaks
from pythonmisc.parallel.parallelgzip import get_files
from fileIO.hdf5.open_h5 import open_h5



def get_shape(fnamelist,framelist=None,group="entry/data/data", troi = None):
    'This function opens the specified hdf5 file at default group = entry/data/data and returns the shape of the data. Includes all related files \n. Default framelist is None (gives all frames), default threshold is none'
    datashape = [0,0,0]
    for fname in fnamelist:
        if not fname.find(".h5") == -1:
            f       = h5py.File(fname, "r")
            datashape[2] += f[group].shape[0]
            # print('datashape is now %s' % datashape)

            if troi == None:
                troi = ((0,0),(f[group].shape[1],f[group].shape[2]))
            datashape[0] = (troi[1][0])
            datashape[1] = (troi[1][1])
            # print('read troi, datashape is now %s' % datashape)
            
        else:
            print("%s is not a .h5 file" %fname)
    if not framelist == None:
        datashape[2] = len(framelist)
        # print('framelist given, datashape is now %s' % datashape)
        
    return datashape

def get_datagroup_shape(datagroup, troi=None, verbose=False):
    '''
    shape of all datasets in datagroup stacked. Usefull for ID13 Eiger master files
    '''
    datashape = [0,0,0]
    datakey_list = datagroup.keys()
    if verbose:
        print('in get_datagroup_shape, got datakey_list: ')
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
    if type(troi)!=type(None):
        datashape[1] = (troi[1][0])
        datashape[2] = (troi[1][1])
    else:
        datashape[1] = dataset.shape[1]
        datashape[2] = dataset.shape[2]
        
    return datashape, datatype

def parse_master_fname(data_fname):
    master_path = os.path.dirname(data_fname)
    master_fname = os.path.basename(data_fname)[:os.path.basename(data_fname).find("data")]+'master.h5'
    return master_path + os.path.sep + master_fname


def parse_data_fname_tpl(master_fname):
    master_path = os.path.dirname(master_fname)
    data_fname_tpl = os.path.basename(master_fname)[:os.path.basename(master_fname).find("master")]+'data_{:06d}.h5'
    return master_path + os.path.sep + data_fname_tpl



def filter_relevant_peaks(data, peaks, verbose = False):
    newpeaks = np.zeros(shape=peaks.shape)

    for l,peak in enumerate(peaks):
        if verbose > 2:
            print("checking peak:")
            print(peak)
        discard  = False

        if peak[0] <= 0:
            discard = True
        if not peak[1] < max(data[0,:]) or not peak[1] > min(data[0,:]):
            discard = True                
        if peak[2] > max(data[1,:]) - min(data[1,:]):
            discard = True
        if not discard:
            newpeaks[l,:] = peak
            if verbose>2:
                print("keeping peak %s:\na%s = %s, mu%s = %s, sigma%s: %s" % (l, l, peak[0], l, peak[1], l, peak[2]))
        else:
            if verbose > 2:
                print("discarding peak %s:\na%s = %s, mu%s = %s, sigma%s: %s" % (l, l, peak[0], l, peak[1], l, peak[2]))

    newpeaks.sort(axis = 0)
    newpeaks = np.flipud(newpeaks)

    newpeaks = summarize_peaks(newpeaks, verbose = verbose)
    
    if verbose:
        print('newpeaks = ')
        print(newpeaks)        
    
    return newpeaks



def summarize_peaks(peaks, verbose = False):
    '''
    expects listnp.array(3,nopeaks) containing sequence  of aX, muX, sigmaX at most nopeaks times, like result of do_multi_gauss_fit in gauss_fitting.py
    returns a similar array with all peaks discarded if they are closer than half the sum of their sigmas together.
    '''

    nopeaks  = len(peaks[:,0])
    
    for k, peak1 in enumerate(peaks[list(range(nopeaks-1)),:]):
        if peak1[0] ==0:
            pass
        else:
            for l, peak2 in enumerate(peaks[k+1:,:]):
                if np.absolute(peak1[1] - peak2[1]) < (peak1[2] + peak1[2]):
                    if verbose:
                        print('found similar peaks:')
                        print(peak1)
                        print(peak2)
                    peak3 = add_peaks(peak1,peak2)
                    peak1 = peak3
                    peak2 *= 0        

    peaks.sort(axis=0)
    peaks = np.flipud(peaks)
    return peaks


def make_eiger_yzth_tpl(eigerfname):
    path         = os.path.dirname(eigerfname) + os.path.sep
    fname        = os.path.basename(eigerfname)  
    partslist    = fname.split('_')[::-1]
    partslist[2] = '%d'
    partslist[3] = '%d'
    
    tpl = path + '_'.join(partslist[::-1])
    return tpl

def make_reference(eigerfname):
    name         = os.path.basename(eigerfname)
    namelist     = name.split('_')[0:-3]
    reference    = "".join(namelist)

    return reference

def get_data_recursive(dictionary,
                       keylist,
                       verbose = False):
    '''
    run down keylist to return final key:
    returns dictionary[key[0]][key[1]]...[key[len(keylist)]]
    '''
    if len(keylist) > 1:
        if verbose:
            print('recursing on key %s' % keylist[0])
        data = get_data_recursive(dictionary[keylist[0]],keylist[1:],verbose = verbose)
    else:
        if verbose:
            print('from dictionary.keys() :')
            print(list(dictionary.keys()))
            print('returning data at key %s' %keylist[0])
#            print(data)
        data = dictionary[keylist[0]]
            
    return data

def get_full_dataset(fname,
                     key = 'i_over_q',
                     verbose=False):
    '''
    searches for all files of the same scan (presuming the prefix is a sufficient criterion) and returns the 'key' dataset stacked in the first dimension -> n times shape (20,2) returns shape (n, 20, 2)
    uses h5_scan (slower, but works for meta dict of h5_scans)
    '''
    from fileIO.hdf5.h5_scan import h5_scan

    if fname.find('.h5')!=-1 and os.path.exists(fname):
        fname = os.path.realpath(fname)
    
    path         = os.path.dirname(fname)
    pathfilelist = get_files(path)
    todolist     = []
    
    reference    = make_reference(fname)
    
    for filename in pathfilelist:
        if make_reference(filename) == reference:
            todolist.append(os.path.realpath(filename))

    todolist.sort()
    keylist = key.split('/')
    print(todolist)
    scan    = h5_scan()
    scan.read_self(todolist[0])
    onedata = get_data_recursive(scan.data, keylist, verbose = verbose)

    newshape = [len(todolist)]
    newshape.extend(onedata.shape)

    if verbose:
        print('reading %s files: ' % len(todolist))
        print(todolist)
        print('keys to find:')
        print(keylist)
        print('found data.shape = ')
        print(onedata.shape)
    
    data    = np.zeros(shape = newshape)
    
    for i,filename in enumerate(todolist):
        scan    = h5_scan()
        scan.read_self(filename)
        onedata = get_data_recursive(scan.data, keylist, verbose =verbose)
        data[i] = onedata
    
    return data

def open_full_dataset(fname,
                     key = 'i_over_q',
                     verbose=False):
    '''
    searches for all files of the same scan (presuming the prefix is a sufficient criterion) and returns the 'key' dataset stacked in the first dimension -> n times shape (20,2) returns shape (n, 20, 2)
        uses open_h5.py
    '''

    if fname.find('.h5')!=-1 and os.path.exists(fname):
        fname = os.path.realpath(fname)
    
    path         = os.path.dirname(fname)
    pathfilelist = get_files(path)
    todolist     = []
    
    reference    = make_reference(fname)
    
    for filename in pathfilelist:
        if make_reference(filename) == reference:
            todolist.append(os.path.realpath(filename))

    todolist.sort()
    onedata = open_h5(todolist[0], group = 'entry/data/'+key)

    newshape = [len(todolist)]

    newshape.extend(onedata.shape)


    print('newshape:')
    print(newshape)

    if verbose:
        print('reading %s files: ' % len(todolist))
        print(todolist)
        print('group to find:')
        print(key)
        print('found data.shape = ')
        print(onedata.shape)
    
    data    = np.zeros(shape = newshape)
    
    for i,filename in enumerate(todolist):
        onedata = open_h5(filename, group = 'entry/data/'+key)
        data[i] = onedata
    
    return data

        
def get_eigerrunno(master_fname):
    '''
    parsed from self.master_fname
    '''

    eiger_runno = int(master_fname.split('_')[-2])
    return eiger_runno


def get_r3_i_list(r3_compatible_path=None):
    '''
    on 23.06.28
    this returns the [[eigerprefix, eiger runno, spec_scanno, and meshshape] ...]
    '''
    tmp_fname  = 'delete_me_{}.tmp'.format(os.getpid())
    r3_lines = check_output('i',shell=True).split('\n')
    i_list=[]
    for line in r3_lines[1:]:
        line_split = line.split()
        if len(line_split)==11:
            i_list.append([line_split[0],int(line_split[1]), int(line_split[-1]), [int(line_split[-4][:-1]),int(line_split[-3][:-1])]])
    
    return i_list

    
