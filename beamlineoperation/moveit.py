from __future__ import print_function
import os
from os import path
from commands import getstatusoutput

TPL = "mv raw_s2426_4_02_%1d_%04d.edf run_%s/"

def move_no(i, n):
    dname = "run_%1d" % i
    try:
        os.mkdir(dname)
    except:
        try:
            print(dname, ":", path.exists(dname))
        except:
            print("fundamental error:", dname)
    for j in xrange(n):
        print("cycle:", j)
        cmd = TPL % (i,j,i)
        print(cmd)
        #raw_input("...")
        print(getstatusoutput(cmd))

def test():
    #move_no(110, 17061)
    for k in range(85,110):
        move_no(k, 17061)

if __name__ == '__main__':
    test()
