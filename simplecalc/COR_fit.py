from scipy.optimize import leastsq
import numpy as np      
from matplotlib import pyplot as plt
# This module is based on the old tolman script


def _f(x, p):
    _x2 = x*x
    return p[0] + p[1]*x + p[2]*_x2

def _rf(p, y, x):
    _x2 = x*x
    #print p, y, x
    return y - (p[0] + p[1]*x + p[2]*_x2)

def _rfsin(p, y, x):
    return y - _fsin(x, p)

def _fsin(x, p):
    #print "fsin x arrg:"
    #print x
    A,ph,o = tuple(p)
#    print "A,Ph,o=", A,ph,o
    return A*np.sin(x-ph) + o

class SinFit(object):

    def __init__(self, ps, pstep, npoints):
        self.ps = ps
        self.pe = ps + (npoints-1)*pstep
        self.xrang = (pe-ps)

def COR_from_yth(yth):
    '''
    expects a np.array with the centered y values for a given Theta (in degrees)
    returns the COR in units of x
    '''
    XPP = np.linspace(-180.0,360.0,67)
    XP  = XPP*np.pi/180.0
    
    th  = np.array(yth[:,1])*np.pi/180.0
    y   = np.array(yth[:,0])
    p   = np.array([5.0,-10.0,0.0], np.float64)
    pl  = leastsq(_rfsin, p, args=(y, th))
    #    print "pl=", pl
    yl  = _fsin(XP, pl[0])
    A   = pl[0][0]
    ph  = pl[0][1]
    o   = pl[0][2]
    COR = np.array([-A*np.cos(ph),A*np.sin(ph)])
    thplot = th*180.0/np.pi

    plt.ion()
    plt.plot(thplot,y, 'bx',label ='y-values')
    plt.plot(XPP,yl,color='b')
    plt.show()
    
    return COR
        
def COR_from_xth(xth):
    '''
    expects a nparray with the centered x values for a given Theta (in degrees)
    returns the COR in units of x
    '''
    XPP = np.linspace(-180.0,360.0,67)
    XP = XPP*np.pi/180.0
    
    th = np.array(xth[:,1])*np.pi/180.0
    x = np.array(xth[:,0])
    p = np.array([5.0,-10.0,0.0], np.float64)
    pl = leastsq(_rfsin, p, args=(x, th))
    # print "pl=", pl
    yl = _fsin(XP, pl[0])
    A = pl[0][0]
    ph = pl[0][1]
    o = pl[0][2]
    COR = np.array([A*np.sin(ph),A*np.cos(ph)])
    thplot = th*180.0/np.pi

    plt.ion()
    fig1, ax1 = plt.subplots(1)
    ax1.plot(thplot,x, 'rx',label = 'x-values')
    ax1.plot(XPP,yl, color='r')
    plt.show()
    return COR
