from __future__ import print_function
import os
from os import path
from commands import getstatusoutput
import subprocess
import time


def move_no(n,k):
    dname     = "./run_%04d"
    TPL       = "find MG154_fluoXAS*_%04d_0* | xargs mv -t ./run_%04d"
    scantime  = 0.001
    
    for j in range(n,k):
        try:
            os.mkdir(dname % j)
        except:
            try:
                print(dname, ":", path.exists(dname % j))
            except:
                print("fundamental error:", dname % j)

        print("moving scan TPL :")
        print(TPL % (j,j))
        #raw_input("...")
        print(os.system(TPL %(j,j)))
        time.sleep(scantime)
        
def test():
    #move_no(110, 17061)
    for k in range(85,110):
        move_no(k, 17061)

if __name__ == '__main__':
    move_no(38,100)
