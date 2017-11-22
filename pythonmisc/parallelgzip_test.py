from __future__ import print_function
# use to gzip files in place using more than one process
# AJ 07.2016

from builtins import input
from builtins import str
from builtins import range
import os
from multiprocessing import Pool
import time
import sys
import subprocess
import shlex

# for task default value:
arg            = "-v9"

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
    print("runs one process per folder")
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
#args [0] = [file to zip]
               
    fpath = inargs
    args =[]
    args.append("gzip")
    args.append(arg)
    args.append(fpath)

    #print "doing: %s in process %s" % (" ".join(args),os.getpid())
    
    subprocess.call(args)

def task2(inargs):
# args[0]= path, args [1] = [file to zip]
               
    fpath = os.path.sep.join(inargs)
    args =[]
    args.append("gzip")
    args.append(arg)
    args.append(fpath)

    #print "doing: %s in process %s" % (" ".join(args),os.getpid())
    
    subprocess.call(args)

def gzip(path):
# comarison for benchmarking

    fpath = path
    args =[]
    args.append("gzip")
    args.append("-rq9")
    args.append(fpath)    
    subprocess.call(args)


def paragzip2(path, noprocesses, arg):
# test with os.walk faster/more stable than find -d with running pools
    print('Creating pool with %d processes\n' % noprocesses)
  
    pool = Pool(processes=noprocesses, maxtasksperchild = 10)
    
    for dirName, subdirList, fileList in os.walk(path):

        print('Found directory: %s' % dirName)
        todolist = []
        [todolist.append([dirName,fname]) for fname in fileList]
        pool.map(task2, todolist, chunksize=10)

    pool.close()
    pool.join()
        

def paragzip(path, noprocesses, arg):
# version with find -d get_files and get_folders
# misteriously sometimes hangs itself on 
# allfiles  = shlex.split(subprocess.check_output(arg))
# in get_files

    print('Creating pool with %d processes\n' % noprocesses)
    
    print('finding list of folders (this may take a while)')
    folders  = get_folders(path)
    todolist = []

#    print todolist    


    for path in folders:
        pool = Pool(processes=noprocesses, maxtasksperchild = 10)
        print("folder : %s" % path)
        filelist = get_files(path)
        pool.map(task, filelist, chunksize=10)
  
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

    

# timing benchmarking
    gtime   = []
    twotime = []
    onetime = []
    for i in range(10):
        print("cycle %s" % i)

        time_one = time.time()        
        
        gzip(path)

        time_two = time.time()

        paragzip2(path, i+1, arg)

        time_three= time.time()

        paragzip(path, i+1, arg)    

        time_four= time.time()
        
        gtime.append(str(float(time_two) - float(time_one)))
        twotime.append(str(float(time_three) - float(time_two)))
        onetime.append(str(float(time_four)  - float(time_three)))

    print("gzip -rv9 took %s s"  % ", ".join(gtime))
    print("paragzip2 took %s s"  % ", ".join(twotime))
    print("paragzip took %s s"   % ", ".join(onetime))



    print("finished in %s" % path)

