from __future__ import print_function
# this needs scipy.version.version >18.1 it runs on 
# source /data/id13/inhouse6/COMMON_DEVELOP/py_andreas/aj_venv/bin/activate
import sys, os
import scipy.spatial as spatial
import numpy as np
import h5py
from multiprocessing import Pool


def remove_close_peaks(data, min_distance):
    '''
    data = nparray 
       with data.shape = (number_of_peaks,3), 
       where data data[:,0] = x, data[:,1] = y and data[:,2] = intensity  
       returns zeros for peaks that were discarded because they were closer than <min_distance> to a more intense peak
    '''
    # sort the peaks according to intensity:
    data = data[data[:,2].argsort()[::-1],:]
    i = 0
    any_left = True
    while any_left and i < data.shape[0]-1:
        # resort by intensity
        checkdata = data[i:,:][data[i:,2].argsort()[::-1],:]    
        # save the winner:
        data[i,:] = checkdata[0,:]
        # check wether there are sensible peaks left:
        if checkdata[1,2]==0:
            any_left = False

        points = zip(checkdata[:,0],checkdata[:,1])
        tree  = spatial.cKDTree(points[1::])
        
        # finds points closer than min_distance to most intense peak at points[0]
        close = tree.query_ball_point(x = points[0], r = min_distance)
        close = [index + 1 for index in close] # because I searched [points[1::]]
        # zeros these peaks:
        checkdata[close,:] = np.zeros(shape = (len(close),3))
        data[i:,:] = checkdata
        i += 1

        # # # debug plotting:
        # import matplotlib.pyplot as plt
        # print 'checking area around ', points[0]
        # print 'throwing out: '
        # print np.asarray(points)[close,:]
        # for point in np.asarray(points)[close,:]:
        #     print 'distance = ', np.sqrt((points[0][0]-point[0])**2 + (points[0][1]-point[1])**2)

        # ax = plt.gca()
        # circle1 = plt.Circle((points[0][0], points[0][1]), radius = min_distance, color='r', fill=False)
        # ax.add_artist(circle1)
        # for point in points[1:]:
        #     print point
        #     circle2 = plt.Circle((point[0], point[1]), radius= 10, color='b', fill=False)
        #     ax.add_artist(circle2)
        # plt.plot(np.asarray(points)[close,0],np.asarray(points)[close,1],'rx')
    
    data[i:,:] = 0
    return data

def _remove_peaks_on_file_level(inargs):
        fname = inargs[0] 
        min_distance = inargs[1] 
        data_path = inargs[2]
        save_path = inargs[3]
    
        # print 'fname ', fname
        readfile     = data_path + fname
        # print 'readfile ', readfile
        readfile_rel = os.path.relpath(data_path,save_path) + os.path.sep + fname
        # print 'readfile_rel ', readfile_rel
        writefile    = save_path + fname
        # print 'writefile ', writefile
        
        # getting data input
        r = h5py.File(readfile,'r')
        print('reading file ', readfile)
        peaks = np.asarray([np.asarray(r['peakXPosRaw']),np.asarray(r['peakYPosRaw']),np.asarray(r['peakTotalIntensity'])])

        # just the peaks, leave out the zeros
        valpeaks = [peaks[:,i,np.where(peaks[0,i,:]>0)][:,0,:] for i in range(peaks.shape[1])]
        
        chosen_peaks = []
        for frame in valpeaks:
            data = np.rollaxis(frame,1) 
            data = remove_close_peaks(data, min_distance)
            frame = np.rollaxis(data, 1)
            # again, just the peaks, leave out the zeros
            frame = frame[:,np.where(frame[2,:]>0)][:,0,:]
            chosen_peaks.append(frame)

        # counting the peaks and adding back the zeros to pad the array up to original size:
        newpeaks = np.zeros(shape=peaks.shape)
        nPeaks   = []  
        for i, peak_array in enumerate(chosen_peaks):
            nPeaks.append(peak_array.shape[1])
            newpeaks[:,i,0:peak_array.shape[1]]+=peak_array

        ## output
        f = h5py.File(writefile,'w')

        # recommendation: only relative paths in links (avoid /gz or /hz mounting problems)
        f['data'] = h5py.ExternalLink(readfile_rel, '/data')
        f.create_dataset(data=np.asarray(nPeaks), name ='nPeaks')
        f.create_dataset(data=newpeaks[0]       , name ='peakXPosRaw')
        f.create_dataset(data=newpeaks[1]       , name ='peakYPosRaw')
        f.create_dataset(data=newpeaks[2]       , name ='peakTotalIntensity')

        f.flush()
        f.close()
        r.close()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        if len(sys.argv)>2:
            min_distance = int(sys.argv[2])
        else:
            print('minimum distance defaults to 30')
            min_distance = 30
        if len(sys.argv)>3:
            noprocesses = int(sys.argv[3])
        else:
            print('number of processes defaults to 4')
            noprocesses= 4

    else:
        print('please specify a npc dataoutput directory')
        sys.exit()
    print('data_path ', data_path)
    data_path_list = data_path.split(os.path.sep)
    # print 'data_path_list', data_path_list
    save_path_list = list(data_path_list)
    # print 'save_path_list[-2]', save_path_list[-2]
    save_path_list[-2] = save_path_list[-2] + '_min%sfiltered/' % min_distance
    save_path = os.path.sep.join(save_path_list)
    print('save_path ', save_path)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fnames = [fname for fname in os.listdir(data_path) if os.path.isfile(data_path+fname)]
    # print 'fnames[:10] ', fnames[:10]


    
    task_list = [[fname, min_distance, data_path, save_path] for fname in fnames]
    print(task_list[:10])

    #for task in task_list:
    #    _remove_peaks_on_file_level(task)

    pool = Pool(processes=noprocesses)
    pool.map(_remove_peaks_on_file_level, task_list)
    
    pool.close()
    pool.join()
