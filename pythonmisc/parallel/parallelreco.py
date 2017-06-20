# use to gzipreconstruct (with convert16.py) files in place using more than one process
# AJ 07.2016
# not finished

import os
from multiprocessing import Pool
import time
import random
import sys
import subprocess
import shlex

### old functions:

def get_folders(src):
# sorted list of folders in src
# from syncodisk

    arg       = []
    arg.append("find")
    arg.append(src)
    arg.append("-type")
    arg.append("d")

    allfolders=shlex.split(subprocess.check_output(arg))
    allfolders.sort()

    return allfolders


def get_files(src,find=None):
# list of files in src (not subfolders)
# adapted from syncodisk

    arg       = []
    arg.append("find")
    arg.append(src + os.path.sep)
    arg.append("-maxdepth")
    arg.append("1")
    arg.append("-type")
    arg.append("f")
    if find != None:
        arg.append(find)
    
#    cmd       = " ".join(arg)
#    print cmd
    allfiles  = shlex.split(subprocess.check_output(arg))
#    print cmd + "\n    worked \n  it seems"
    return allfiles


### new stuff

def usage():
    print "python parrallelreco.py <no of processes (default 4)> <path>"
    print "runs one process per file"
    sys.exit(0)

def confirm(path, noprocesses, arg):
# yes or exit
    prompt = "Do you want to reconstruct %s with %s parallel processes with these arguements: \n [y/n] " 
    if raw_input(prompt % (path, noprocesses)) in ("y","yes"):
        print "will do"
    else:
        print "ok, quitting"
        sys.exit(0)
    


def task(path):

    args =[]
    args.append("python")
    args.append("convert16.py")
#    args.append("-v9")
#    args.append(path)
#    args.append(path)               
    print "doing: %s in process %s" % (path,os.getpid())

    flist  =  get_files(path,"*.edf")

    for fname in flist:
        args.append(fname)
    
    subprocess.call(args)




def setup_pool(path, noprocesses):

    print 'Creating pool with %d processes\n' % noprocesses
    pool = Pool(processes=noprocesses)
    
    if type(path)=list:
        folders = path
    else:
        folders  = get_folders(path)    
    
    pool.map(task,folders)
 



if __name__ == '__main__':
 
#    multiprocessing.freeze_support()
#   default values:
    noprocesses    = 4
    rest = []

    if len(sys.argv) == 2:
        path = str(sys.argv[1])
    elif len(sys.argv) > 2:
        path           = []
        args  = sys.argv[1:]
        for x in args:
            try:
                noprocesses = int(x)
            except KeyError:
                path.append(x)       
    else:
        usage()


    print 'Using %d processes to reconstruct all folders in \n%s\n in parallel' % (noprocesses,"\n".join(path))
 
    print get_files(path)

    confirm("\n".join(path), noprocesses, arg)
    paragzip(path, noprocesses, arg)
