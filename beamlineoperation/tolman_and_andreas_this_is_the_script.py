from __future__ import print_function
from builtins import input
from builtins import str
from builtins import object
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from o8x3.io import xyio

import numpy as np

def readxy(fname, dname=None, skip=0):
    if dname:
        fname = os.path.join(dname, fname)
    f = file(fname, "r")
    ll = f.readlines()
    print("hi debugger")
    print(ll)
    f.close()
    ll = ll[skip:]
#    print(ll)
    x = np.zeros((len(ll),), dtype=np.float64)
    y = np.zeros((len(ll),), dtype=np.float64)
    for (i,l) in enumerate(ll):
        try:
            (x[i],y[i]) = (float(l.split()[0]),float(l.split()[1]))
            
        except IndexError as bla:
            print(bla)
    return (x,y)


def f(x, p):
    _x2 = x*x
    return p[0] + p[1]*x + p[2]*_x2

def rf(p, y, x):
    _x2 = x*x
    print(p, y, x)
    return y - (p[0] + p[1]*x + p[2]*_x2)

def rfsin(p, y, x):
    return y - fsin(x, p)

def fsin(x, p):
    print("fsin x arrg:")
    print(x)
    A,ph,o = tuple(p)
#    print "A,Ph,o=", A,ph,o
    return A*np.sin(x-ph) + o

class SinFit(object):

    def __init__(self, ps, pstep, npoints):
        self.ps = ps
        self.pe = ps + (npoints-1)*pstep
        self.xrang = (pe-ps)

XPP = np.linspace(-180.0,180.0,37)
XP = XPP*np.pi/180.0

def _test():

    (x,y) = readxy("/data/id13/inhouse6/THEDATA_I6_1/d_2017-01-26_inh_ihhc3060/DATA/R3_new/COR.txt") 
    
    x = np.array(x)*np.pi/180.0
    y = np.array(y)
    p = np.array([5.0,-10.0,0.0], np.float64)
    pl = leastsq(rfsin, p, args=(y, x))
    print("pl=", pl)
    yl = fsin(XP, pl[0])
    A = pl[0][0]
    ph = pl[0][1]
    o = pl[0][2]
    print("phase=",  divmod(pl[0][1]*180.0/np.pi, 360.0))
    print("y-comp =", -A*np.sin(ph))
    print("x-comp =", A*np.cos(ph))
    print("That means: umvr nnx " +  str(A*np.cos(ph)) +" X  " + str(-A*np.cos(ph)) + " nny " +  str(-A*np.sin(ph)) + " Y " + str(A*np.sin(ph)))

    xplot = x*180.0/np.pi
    plt.ion()
    plt.plot(xplot,y)
    plt.plot(XPP,yl)
    plt.draw()
    input()

if __name__ == '__main__':
    _test()
    
