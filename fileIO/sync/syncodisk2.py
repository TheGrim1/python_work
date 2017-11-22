from __future__ import print_function
import os
import sys


## local import 
import syncotools as sync
from syncotools import part


def printusage():

    print('Usage: \npython syncodisk.py -<mode> <sourcepath> <destitation path of part1> <size of part1 in bytes> <path of part2> etc.')
    print('\nTry to keep foldersizes well below half of your max disk size for optimal results')
    print('\nto repeat only -<mode> and <sourcepath> (if different from current dir) are required')
    print('\nIf "t" in <mode> - test mode\n  - rsync will not write anything')
    print('\nIf "n" in <mode> - new run mode\n  - delete previous allocations. Note: no data files are deleted, previously transferred data my be duplicated to different parts.')
    print('\nIf "d" in <mode> - default mode\n  - repeat allocations as found in <source dir> or make new allocation. Note: no data files are deleted, previously transferred data my be duplicated to different parts (but should not).')
    print('e.g. \n "python syncodisk.py -nt /data/id13/inhouse2/AJ/skript/fileIO/test /data/id13/inhouse2/AJ/skript/fileIO/disc0/ 200 /data/id13/inhouse2/AJ/skript/fileIO/disc1 200"')
    sys.exit(0)
        

def sort_user_input(userargv):

    print("")
    print("")

    argv=[]
    userpaths = [] 
    mode = " "
    for arg in userargv:
        if arg.find(os.path.sep)!=-1 or arg.isdigit():
            argv.append(arg)
        if arg.find("-")!= -1 and arg.find("/")==-1 :
            mode = arg
            
    if len(argv)>0:
        src = argv[0]
    else:
        src = os.getcwd()

    if mode.find("n")!=-1:
        deletefilter(src)
            
    if mode == "-t":
        mode = "-tm"
    if len(mode) == 1:
        mode = "-m"

    if mode.find(d) != -1:
        mode = "-m" 

    try:
        src  = os.path.realpath(argv[0])
    except IndexError: 
        src  = os.path.realpath(os.getcwd())


    return (src, mode, argv)

def define_destpaths(parts, mode, argv):
# sort out how the user gave this info and df if neccessary
    
    destpaths = []
    if mode.find("m")==-1: 
        if len(argv) >= len(parts):
            printusage()
            sys.exit(1)
        if len(parts) <= 1:
            printusage()
            sys.exit(1)
        for part in parts:
            destpaths.append([part.partpath,part.writesize])

    if mode.find("m")!=-1:
        try:
            for i in range(1,len(argv),2):
                try:
                    destpaths.append([os.path.realpath(argv[i]),argv[i+1]])
                except IndexError:
                    print("I did not understand path %s /n with size %s" % (os.path.realpath(argv[i]),argv[i+1]))
                    printusage()
        except IndexError:
            printusage()

    return destpaths

    
def initiate_parts(src, mode, destpaths, parts):

    i = 0
    for pathinfo in destpaths:

        found = False
        for part in parts:
            if os.path.realpath(part.partpath) == os.path.realpath(pathinfo[0]):
                print("found path %s in part %s" % (part.partpath, part.partno))
                part.writesize = pathinfo[1]
                found = True
        if not found:
            print("creating new part no %s written to path %s" %(i, pathinfo[0]))
            parts.append(sync.create_newpart(src, pathinfo,i))

        i =+ 1

    pathtree          ={}

    print("Scanning source folder %s\n ... (this can take a while) " % src)

    pathtree          = sync.scansource(src)

#    sync.dump(pathtree)      
#    print " debug "
#    sys.exit(1)
        
# Not finished:
#    if mode.find("s")!=-1:
#        for i in range(len(parts)):
#            print "Scanning part%s in %s for folders allready copied ..." % (i, parts[i].partpath)
#            parts[i] = scanpart(parts[i],pathtree)
#            dump(parts[i].dirs)

    print("Allocating folders from source to parts ... ")
    (parts, pathtree) = sync.findsplit(parts,pathtree)
 

    tbwsize = 0
    for i in range(len(parts)):
        tbwsize+=parts[i].writesize
        sync.save(parts[i])

    sync.printparts(parts)

    print("total number of bytes found:              %s" % pathtree[src]["info"]["branchsize"])
    print("total number of bytes allocated to parts: %s" % tbwsize)
    print("-------------------------------------------------")

    return parts



def main(userargv):

    (src, mode, argv) = sort_user_input(userargv)

    parts     = sync.findfilter(src)
    
    # part is a class collecting all the information to be a ble to run a sync saved in partX.txt

    destpaths = define_destpaths(parts, mode, argv)
    
    # destpaths is a list with a list containing a path and a size in bytes to be written onto that path

    parts = initiate_parts(src, mode, destpaths, parts)
 
    sync.doall(parts,mode)

if __name__ == '__main__':
    
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc. 
\n2) python <thisfile.py> -f <file containing args as lines> 
\n3) find <*yoursearch* -> arg1 etc.> | python <thisfile.py> 
"""

    args = []
    if len(sys.argv) > 1:
        if sys.argv[1].find("-f")!= -1:
            f = open(sys.argv[2]) 
            for line in f:
                args.append(line.rstrip())
        else:
            args=sys.argv[1:]
    else:
        printusage()
    
#    print args
    main(args)
    sys.exit(0)
