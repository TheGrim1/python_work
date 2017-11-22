from __future__ import print_function
from builtins import range
import sys
import os
import ast

def reformat_calibline(caliblines = ["Calibration done at [2016-10-25 Tue 13:59]\n",
                                     "| [PixelSize1, PixelSize2, Distance, Poni1, Poni2, Rot1, Rot2, Rot3] | [7.5e-5, 7.5e-5, 0.12269051, 0.0933, 0.17910368, 0, 0, 0] |"]):
    
    datalist = caliblines[1].split("|")[1:3]

    datalist[0] = datalist[0].lstrip().lstrip("[").rstrip().rstrip("]").split(",")
    datalist[1] = datalist[1].lstrip().lstrip("[").rstrip().rstrip("]").split(",")

    poniline = [caliblines[0]]
    for i in range(len(datalist[1])):
        poniline.append(datalist[0][i].lstrip() + ": " + datalist[1][i].lstrip()+ "\n")

    return poniline

def read_calibtext(fname):
   
    poni = "nothing happened"
    #                print("reading %s " % os.path.join(fname))
    f = open(fname,"r")
    caliblist=f.readlines()
    if len(caliblist)!=2:
        print("ERROR on %s \nthis file has not got exactly two lines" % fname)
    else:
        poni = reformat_calibline(caliblist)
    f.close()


    return poni


def main(args):

    for fname in args:
        if fname.endswith(".txt") and fname.find("calib")!=-1:    
            print("poni for file %s" % fname)
            
            poniline = read_calibtext(os.path.realpath(fname))
            print(poniline)
        
            g = open(fname.rstrip(".txt")+".poni","w")
            g.writelines("%s" % l for l in poniline)


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
        f = sys.stdin
        for line in f:
            args.append(line.rstrip())
    
#    print args
    main(args)
