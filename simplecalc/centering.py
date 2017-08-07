from matplotlib import pyplot as plt
from scipy.optimize import leastsq
import numpy as np
import scipy.ndimage as nd
import sys,os
# local imports
path_list = os.path.dirname(__file__).split(os.path.sep)
importpath_list = []
if 'skript' in path_list:
    for folder in path_list:
        importpath_list.append(folder)
        if folder == 'skript':
            break
importpath = os.path.sep.join(importpath_list)
sys.path.append(importpath)        
from simplecalc import image_align as ia
from simplecalc import rotated_series as rs


        
def f(x, p):
    _x2 = x*x
    return p[0] + p[1]*x + p[2]*_x2

def rf(p, y, x):
    _x2 = x*x
    #print p, y, x
    return y - (p[0] + p[1]*x + p[2]*_x2)

def rfsin(p, y, x):
    return y - fsin(x, p)

def fsin(x, p):
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
    expects a nparray with the centered x values for a given Theta (in degrees)
    returns the COR in units of x
    '''
    XPP = np.linspace(-180.0,360.0,67)
    XP  = XPP*np.pi/180.0
    
    th  = np.array(yth[:,1])*np.pi/180.0
    y   = np.array(yth[:,0])
    p   = np.array([5.0,-10.0,0.0], np.float64)
    pl  = leastsq(rfsin, p, args=(y, th))
    #    print "pl=", pl
    yl  = fsin(XP, pl[0])
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
    pl = leastsq(rfsin, p, args=(x, th))
    # print "pl=", pl
    yl = fsin(XP, pl[0])
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

def COR_2d_crosscorrelation(imagestack,thetas, rotation = 0):
    '''
    returns the cordinates of the COR in imagestack[0,:,:]
    rotation = aditional angle of the projection in xy plane in degrees
    '''
    dummy        = np.copy(imagestack)
    center       = 0.5*np.array(dummy[0].shape)
    thetas       = np.array(thetas) + rotation

    ### before crosscorrelating, the images have to be rotated back:
    do_thetas = (-1*thetas)
    dummy = rs.rotate_series(dummy,
                             do_thetas,
                             COR = center,
                             copy =False)
        
    dummy, shift = ia.crosscorrelation_align(dummy)
    
    xth = [(x,do_thetas[i]) for i,(x,y) in enumerate(shift)]
    yth = [(y,do_thetas[i]) for i,(x,y) in enumerate(shift)]
    xth = np.asarray(xth)
    yth = np.asarray(yth)

    print 'center:   ', center
    CORx      = center - np.array(COR_from_xth(xth))
    CORy      = center - np.array(COR_from_yth(yth))
    COR_found = 0.5*(CORx + CORy)
    print 'COR before backprojection:' 
    print 'fit to x-values:  ', CORx
    print 'fit to y-values:  ', CORy
    print 'average:          ', COR_found
    
    ## this seems to be the right backprojection:
    COR       = 2*center - COR_found
    print 'center of rotation found:'
    print COR


    return COR

def COR_1d_COM(linestack, thetas, rotation=0):
    '''
    return COR 
    lines stacked in linestack.shape[0]
    lines linestack are projections along 'y', i.e. linestack.shape[1] is x for projection = 0
    rotation = angle of the projection in xy plane in degrees
    '''
    dummy        = np.copy(linestack)
    dummy, shift = ia.centerofmass_align(dummy)
    COM          = nd.measurements.center_of_mass(dummy[0])

    thetas = np.array(thetas) + rotation
    
    xth = [(x[0],thetas[i]) for i,x in enumerate(shift)]
    xth = np.asarray(xth)


    print 'COM in first frame:   ', COM
    COR   = COM - np.array(COR_from_xth(xth))[0]
    print 'center of rotation' 
    print 'fit to x-values:  ', COR
    
    return COR

def COR_2d_COM(imagestack,thetas, rotation = 0):
    '''
    returns the cordinates of the COR in imagestack[0,:,:]
    rotation = aditional angle of the projection in xy plane in degrees
    '''
    dummy        = np.copy(imagestack)
    dummy, shift = ia.centerofmass_align(dummy)
    COM          = nd.measurements.center_of_mass(dummy[0,:,:])
    thetas       = np.array(thetas) + rotation
    
    xth = [(x,thetas[i]) for i,(x,y) in enumerate(shift)]
    yth = [(y,thetas[i]) for i,(x,y) in enumerate(shift)]
    xth=np.asarray(xth)
    yth=np.asarray(yth)

    print 'COM:   ', COM
    CORx   = COM - np.array(COR_from_xth(xth))
    CORy   = COM - np.array(COR_from_yth(yth))
    COR    = 0.5*(CORx + CORy)
    print 'center of rotation' 
    print 'fit to x-values:  ', CORx
    print 'fit to y-values:  ', CORy
    print 'average:          ', COR
    COR = COR
    
    return COR

def align_COR(imagestack,thetas,COR):
    '''
    alignes imagestack according the given thetas and COR
    '''
    do_thetas     = -1*np.asarray(thetas)
    dummy         = np.copy(imagestack)
    aligned_stack = rs.rotate_series(dummy, do_thetas, COR)    
    return aligned_stack


def COR_from_imagestack(imagestack, thetas, mode = '2D_COM', align = True):
    '''
    if align = True, aligns image_stack in place
    properies['modes'] available:
    '2d_COM', '1D_COM', '2D_crosscorrelation'
    '''
    if mode.upper() == '2D_COM':
        COR = COR_2d_COM(imagestack,thetas)
    elif mode.upper() == '1D_COM': 
        COR = COR_1d_COM(imagestack,thetas)
    elif mode.upper() == '2D_CROSSCORRELATION':
        COR = COR_2d_crosscorrelation(imagestack,thetas)
    else:
        print 'available modes are: \n1D_COM, 2D_COM, 2D_crosscorrelation'
        raise NotImplementedError('mode %s is not implemented' %mode)
        
    if align:
        aligned_stack = align_COR(imagestack, thetas, COR)
            
    return aligned_stack, COR

def test():
    '''
    illustrates the functionality of this module
    '''
    import matplotlib.pyplot as plt

    array           = np.zeros(shape = (20,30))
    array[12:15,17] = 1
    array[13,16:19] = 1
    array[13,17]    = 4
    COR             = np.array((11,13))
    #thetas          = [x*360/360.0 for x in range(360)]
    thetas          = [0,5,15,42,170,250,255,100,90]
    
    plt.ion()
    fig1, ax1 = plt.subplots(1)
    ax1.matshow(array)
    ax1.plot(COR[0],COR[1],'rx')
    ax1.set_title('original array, set COR = red')
    fig1.show()

    
    rotated = rs.create_rotated_series(array, thetas,COR=COR)

    fig2, ax2 = plt.subplots(1)
    ax2.matshow(rotated.sum(0))
    ax2.plot(COR[1],COR[0],'rx')
    ax2.set_title('summed dataset, rotated by thetas, set COR = red')

    aligned, COR_found = COR_from_imagestack(rotated, thetas)

    fig3, ax3 = plt.subplots(1)
    ax3.matshow(aligned.sum(0))
    ax3.plot(COR_found[1],COR_found[0],'go')    
    ax3.plot(COR[1],COR[0],'rx')
    ax3.set_title('summed, realigned dataset, set COR = red, found COR green')
    
    print 'given COR: ', COR
    print 'found COR: ', COR_found
    
    return True
