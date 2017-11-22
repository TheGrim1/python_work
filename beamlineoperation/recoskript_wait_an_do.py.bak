import os
from commands import getstatusoutput
import time


TPL           = "python convert16.py %s %1d"
waittime      = 3600 # s
eigeroffset   = 274
indices       = ["a","b","c","d","e","g1","g2","g3","g4","h1","h2","h3","h4"]
samplelist    = ["sin002%s_zp" % i for i in indices]
#samplelist    = ["sin002a_zp",
#                 "sin002b_zp",
#                 "sin002c_zp",
#                 "sin002d_zp"]

def reco():
    for j in range(len(samplelist)):
        print "cycle:", j 
        print "waiting %s s" % waittime 
        time.sleep(waittime)

#        internalnum = j
        eigerscannum = j + eigeroffset 

        cmd = TPL % (samplelist[j], eigerscannum)
        print cmd
        #raw_input("...")
        print getstatusoutput(cmd)

if __name__ == '__main__':
    reco()
