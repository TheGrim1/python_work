# use to gzip files in place using more than one process
# AJ 07.2016

import os
from multiprocessing import Pool
import time
import sys
import subprocess
import shlex

# for task default value:
arg            = "-v9"

def usage():
    print "python parrallelgzip.py <path> <no of processes (default 4)>  -<optional gzip arguement>"
    print "runs one process per folder"
    sys.exit(0)

def confirm(path, noprocesses, arg):
# yes or exit
    prompt = "Do you want to run gzip in all of the folders in \n%s\nin %s parallel processes with these arguements: %s\n [y/n] " 
    if raw_input(prompt % (path, noprocesses, arg)) in ("y","yes"):
        print "will do"
    else:
        print "ok, quitting"
        sys.exit(0)

def task2(inargs):
# args[0]= path, args [1] = [file to zip]
               
    fpath = os.path.sep.join(inargs)
    args =[]
    args.append("gzip")
    args.append(arg)
    args.append(fpath)

    #print "doing: %s in process %s" % (" ".join(args),os.getpid())
    
    subprocess.call(args)

def paragzip2(path, noprocesses, arg):
# test with os.walk faster/more stable than find -d with running pools

    print 'Creating pool with %d processes\n' % noprocesses
  
    pool = Pool(processes=noprocesses, maxtasksperchild = 10)
    
    for dirName, subdirList, fileList in os.walk(path):
        print('Found directory: %s' % dirName)
        todolist = []
        [todolist.append([dirName,fname]) for fname in fileList]
        pool.map(task2, todolist, chunksize=10)

    pool.close()
    pool.join()
        


if __name__ == '__main__':
    global arg
#    multiprocessing.freeze_support()
#   default values:
    noprocesses    = 4

    path           = os.getcwd()
    rest = []

    if len(sys.argv) == 1:
        print "default: working in current directory"
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


    print 'Using %d processes to gzip all files in folders in %s in parallel' % (noprocesses,path)
 

    confirm(path, noprocesses, arg)

    timestart = time.time()
    paragzip2(path, noprocesses, arg)
    timetook  = time.time() -timestart

    print "finished in %s\nafter %ss" % path,timetook

