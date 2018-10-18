'''
This is supposed to illustrate the problem ID13 is expiriencing with the performance of bitshuffle
'''
import h5py
import socket
import time
import sys,os
from multiprocessing import Pool
import numpy as np

workstation = socket.gethostname()


NFRAMES = 2000 # up to 2000 are in a file. This should saturate the caches?

if workstation in ['led13gpu1', 'lid13gpu2']:
    PATH = '/hz/data/id13/inhouse3/THEDATA_I3_2/d_2018-09-26_user_ls2873_bianconi/DATA/AUTO-TRANSFER/eiger1/'
else:
    PATH = '/data/id13/inhouse3/THEDATA_I3_2/d_2018-09-26_user_ls2873_bianconi/DATA/AUTO-TRANSFER/eiger1/'

def do_something(frame):
    return frame[slice(0,2000,100),200].sum()

def do_this_file(fname):
    
    arb = 0
    cnt = 0
    with h5py.File(fname,'r') as h5_f:
        print('{} doing {}'.format(os.getpid(),fname))
        for i in range(NFRAMES):
            arb += do_something(h5_f['entry/data/data'][i])
            cnt +=1
            # print(cnt)

    # h5_f.close()
    return arb, cnt


def do_test(arg):
    print('finding files:')
    fname_list = os.listdir(PATH)
    fname_list.sort()
    bslz4_fname_list = [PATH + fname for fname in fname_list if fname.find('21_data')>0][0:80]
    lz4_fname_list = [PATH + fname for fname in fname_list if fname.find('20_data')>0][0:80]

    print('got {} files with bslzf4\nand {} with lzf4 compression'.format(len(bslz4_fname_list),len(lz4_fname_list)))

    print('For 8 data files of each list I will perform sum(frame[slice(0,2000,100),200]) on each frame and see how long this takes')

    for i in range(arg,arg+1):
        
    # bonus question: why does the 2nd loop fail?! eg.:
    # for i in range(arg,arg+2):
    
        bslzf4_list1 = bslz4_fname_list[i*16:i*16+8]
        bslzf4_list2 = bslz4_fname_list[i*16+8:(i+1)*16]
        
        lzf4_list1 = lz4_fname_list[i*16:i*16+8]
        lzf4_list2 = lz4_fname_list[i*16+8:(i+1)*16]
       
    
        print('***'*25)
        print('BITSHUFFLE 8 processes')
        print("hint: do a top in a different terminal to see how many processes I'm hogging")
        start_time=time.time()
        arb_num = 0
        counter = 0

        pool = Pool(8)
        result = pool.map(do_this_file, bslzf4_list1)
        pool.close()
        pool.join()

        arb_num = np.asarray(result)[:,0].sum()
        counter = np.asarray(result)[:,1].sum()
        
        end_time = time.time()
        diff_time = end_time-start_time
        print('I got arbitrary number {}'.format(arb_num))
        print('It took {:.2f}s to read {} frames'.format(diff_time, counter))
        print('           = {:.2f}Hz'.format(counter/diff_time))


        
        print('***'*25)
        print('NO BITSHUFFLE 8 processes')
        print("hint: do a top in a different shell to see how many processes I'm hogging")
        start_time=time.time()
        arb_num = 0
        counter = 0
        
        pool = Pool(8)
        result = pool.map(do_this_file, lzf4_list1)
        pool.close()
        pool.join()
        
        arb_num = np.asarray(result)[:,0].sum()
        counter = np.asarray(result)[:,1].sum()

        end_time=time.time()
        diff_time = end_time-start_time
        print('I got arbitrary number {}'.format(arb_num))
        print('It took {:.2f}s to read {} frames'.format(diff_time, counter))
        print('           = {:.2f}Hz'.format(counter/diff_time))



        
        print('***'*25)
        print('BITSHUFFLE serial')
        print("hint: do a top in a different terminal to see how many processes I'm hogging")
        start_time=time.time()
        
        arb_num = 0
        counter = 0
        for fname in bslzf4_list2:
            res = do_this_file(fname)            
            arb_num += res[0]
            counter += res[1]
            
        end_time=time.time()
        diff_time = end_time-start_time
        print('I got arbitrary number {}'.format(arb_num))
        print('It took {:.2f}s to read {} frames'.format(diff_time, counter))
        print('           = {:.2f}Hz'.format(counter/diff_time))
        

        print('***'*25)
        print('NO BITSHUFFLE serial')
        print("hint: do a top in a different shell to see how many processes I'm hogging")
        start_time=time.time()
        
        arb_num = 0
        counter = 0
        for fname in lzf4_list2:
            res = do_this_file(fname)
            arb_num += res[0]
            counter += res[1]
    
        end_time=time.time()
        diff_time = end_time-start_time
        print('I got arbitrary number {}'.format(arb_num))
        print('It took {:.2f}s to read {} frames'.format(diff_time, counter))
        print('           = {:.2f}Hz'.format(counter/diff_time))

        
if __name__=='__main__':
    arg = int(sys.argv[1])
    do_test(arg)
    
