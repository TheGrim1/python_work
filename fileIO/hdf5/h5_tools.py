from __future__ import print_function
from builtins import range
import h5py
import numpy as np
import sys, os

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
