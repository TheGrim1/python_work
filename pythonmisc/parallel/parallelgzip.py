from __future__ import print_function
# use to gzip files in place using more than one process
# AJ 07.2016

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


def get_files(src):
# list of files in src (not subfolders)
# adapted from syncodisk

#    print "getting files in %s" %src
    arg       = []
    arg.append("find")
    arg.append(src)
    arg.append("-maxdepth")
    arg.append("1")
    arg.append("-type")
    arg.append("f")
#    cmd       = " ".join(arg)
#    print cmd
    allfiles  = shlex.split(subprocess.check_output(arg))
#    print cmd + "\n    worked \n  it seems"
    return allfiles


### new stuff

def usage():
    print("python parrallelgzip.py <path> <no of processes (default 4)>  -<optional gzip arguement>")
    print("runs batches of processes (up to the specified number) in each folder, one process per file")
    sys.exit(0)

def confirm(path, noprocesses, arg):
# yes or exit
    prompt = "Do you want to run gzip in all of the folders in \n%s\nin %s parallel processes with these arguements: %s\n [y/n] " 
    if input(prompt % (path, noprocesses, arg)) in ("y","yes"):
        print("will do")
    else:
        print("ok, quitting")
        sys.exit(0)
    


def task(inargs):
# args[0]= gziparg, args [1] = [file to zip]
               
    fpath = inargs[1]
    arg  = inargs[0]
    args =[]
    args.append("gzip")
    args.append(arg)
    args.append(fpath)

    print("doing: %s in process %s" % (" ".join(args),os.getpid()))
    
    subprocess.call(args)




def paragzip(path, noprocesses, arg):

    print('Creating pool with %d processes\n' % noprocesses)
    
    print('finding list of folders (this may take a while)')
    folders  = get_folders(path)
    todolist = []

#    print todolist    
    pool = Pool(processes=noprocesses)

    for path in folders:
        
        print("folder : %s" % path)
        filelist = get_files(path)
        [todolist.append([arg,x]) for x in filelist]
        pool.map(task,todolist)
  

if __name__ == '__main__':
 
#    multiprocessing.freeze_support()
#   default values:
    noprocesses    = 4
    arg            = "-v9"
    path           = os.getcwd()
    rest = []


    if len(sys.argv) == 1:
        print("default: working in current directory")
    elif len(sys.argv) == 2:
        path = str(sys.argv[1])
    elif len(sys.argv) in  (3,4):
#        print "args given = %s" % sys.argv
        args  = sys.argv[1:]
#        print args
        for x in args:
            if x.find("-")!=-1 and x.find(os.path.sep)==-1:
                arg = x
            else:
                rest.append(x)       
        try:
            path        = str(rest[0])
            noprocesses = int(rest[1])
#            print "processes:  %s,  path:  %s" %(noprocesses,path)
        except KeyError:
            try:
                path        = str(rest[1])
                noprocesses = int(rest[0])
#                print "processes:  %s,  path:  %s" %(noprocesses,path)
            except KeyError:
                usage()
        except IndexError:
            try:
                path = (rest[0])
            except KeyError:
                usage()
    else:
        usage()


    print('Using %d processes to gzip all folders in %s in parallel' % (noprocesses,path))
 

    confirm(path, noprocesses, arg)
    paragzip(path, noprocesses, arg)
    print("finished in %s" % path)
