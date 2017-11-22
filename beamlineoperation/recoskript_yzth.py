from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
import os
from subprocess import getstatusoutput
import sys
import time



def reco_no(args):
    #time.sleep(0)
    #print("sleeping " + initialwait)
    #time.sleep(2)
    print("test") 
    TPL              = "python convert16.py %s_%1d %1d"
    eigerprefix      = args[0]
    start            = int(args[1])
    stop             = int(args[2])
    eigerno          = int(args[3])
    waittime         = float(args[4]) 
    

    initialwait      = 0.1
    #print("sleeping " + initialwait)
    time.sleep(initialwait)
    
    print(list(range(start,stop,2)))
    for j in range(start,stop,2):
        internalnum = j
        eigerscannum = j + eigerno
        print("cycle:", j)
        print("waiting: ",waittime)
        time.sleep(waittime)
        cmd = TPL % (eigerprefix, internalnum, eigerscannum)
        print(cmd)
        #raw_input("...")
        print(getstatusoutput(cmd))

def usage():
    print("usage:\npython recoskript_yzth <eigerprefix> <internal counter start> <stop> <eigeroffset to internal counter> <waittime>")

if __name__ == '__main__':
    if len(sys.argv) != 6:
        usage()
    else:
        reco_no(sys.argv[1:])
