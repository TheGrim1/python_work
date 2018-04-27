import sys,os
import numpy as np
import time

class parallel_timer(object):
    """
    can be used to time independent processes using filebased communicaion
    initialied with a temp directory
    will os.mkdir(parrallel_timer_<random_int>)
    and create a file per process with fname = parallel_timer.newfile()
    the independent external child process should delete the file <fname> when it is finished

    the spawning process can poll parallel.runnning() to check whether all files are deleted
    """

    def __init__(self, basepath):

        np.random.seed(int(time.time()))
        randint = np.random.randint(1000000)
        self.folder = os.path.sep.join([basepath,'parallel_timer_{}'.format(randint)])
        os.mkdir(self.folder)
        self.file_counter = 0
        self.files = []

    def newfile(self):
        self.file_counter +=1
        fname = os.path.sep.join([self.folder,'delete_me_{}.tmp'.format(self.file_counter)])
        f = open(fname,'w')
        f.close()
        self.files.append(fname)
        
        return fname

    def running(self,verbose = False):
        is_running = False
        for fname in self.files:
            f_exists = os.path.exists(fname)
            if f_exists:
                is_running = True

            if verbose:
                print('parallel_timer found file {} - {}'.format(fname,f_exists))
        if not is_running:
            os.rmdir(self.folder)        
        return is_running
        
