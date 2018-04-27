import os
from os import path
from commands import getstatusoutput
import subprocess
import time


def move_no(n,k):
    dname     = "./run_%04d"
    TPL       = "find *_fluoXAS_*_%04d_0* | xargs mv -t ./run_%04d"
    scantime  = 600 
    
    for j in range(n,k):
        try:
            os.mkdir(dname % j)
        except:
            try:
                print dname, ":", path.exists(dname % j)
            except:
                print "fundamental error:", dname % j

        print "moving scan TPL :"
        print TPL % (j,j)
        #raw_input("...")
        print os.system(TPL %(j,j))
        print("sleeping...")
        time.sleep(scantime/3.0)
        print("still sleeping...")
        time.sleep(scantime/3.0)
        print("sleeping a bit more ...")
        time.sleep(scantime/3.0)

        
def test():
    #move_no(110, 17061)
    for k in range(85,110):
        move_no(k, 17061)

if __name__ == '__main__':
    move_no(9,120)
