import os
from os import path
from commands import getstatusoutput
import subprocess
import time


def copy_back_range(n,k):
    source_dname_tpl = "/data/visitor/ma3564/id16b/sample_h/wire_3_c/fluoXAS_0_fluoXAS_0/run_{:04d}/"
    target_dname     = "/data/visitor/ma3564/id16b/analysis/wire_3_c/XRF"
    command_tpl      = "find {}*_fluoXAS_*_{:04d}_0* | xargs mv -t {}"
    
    try:
        os.mkdir(target_dname % j)
    except:
        try:
            print dname, ":", path.exists(dname % j)
        except:
            print "fundamental error:", dname % j

    
    for j in range(n,k):
        
        print("moving scan TPL :")
        cmd = command_tpl.format(source_dname_tpl,j,target_dname)
        print(cmd)
        #raw_input("...")
        print os.system(cmd)

if __name__ == '__main__':
    copy_back_range(9,120)
