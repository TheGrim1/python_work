from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
import os
from os import path
from subprocess import getstatusoutput
import subprocess
import time


def move_no(n,k):
    dname     = "./run_%04d"
    TPL       = ["mv"]
    TPL.append("MG154_fluoXAS*_%04d*")
    TPL.append("./run_%04d")
    scantime  = 1
    
    for j in range(n,k):
        try:
            os.mkdir(dname % j)
        except:
            try:
                print(dname, ":", path.exists(dname % j))
            except:
                print("fundamental error:", dname % j)

        print("moving scan TPL :")
        TPL[1] = TPL[1] % j
        TPL[2] = TPL[2] % j
        print(TPL)
        #raw_input("...")
        print(subprocess.check_output(TPL))
        time.sleep(scantime)
        
def test():
    #move_no(110, 17061)
    for k in range(85,110):
        move_no(k, 17061)

if __name__ == '__main__':
    move_no(0,1)
