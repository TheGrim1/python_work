from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
import os
from subprocess import getstatusoutput

TPL = "python convert16.py vo2_1_xzthscan_%1d %1d"

def reco_no(k,m):
    for j in range(k,m):
        internalnum = j + 1258
        eigerscannum = j + 126
        print("cycle:", j)
        cmd = TPL % (internalnum,eigerscannum)
        print(cmd)
        #raw_input("...")
        print(getstatusoutput(cmd))


def test():
    reco_no(0,17)
    #for k in range(96,110):
     #   move_no(k, 17061)

if __name__ == '__main__':
    test()
