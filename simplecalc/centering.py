from __future__ import print_function
from __future__ import absolute_import
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
import numpy as np
import scipy.ndimage as nd
import sys,os
import warnings
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
from .COR_fit import COR_from_yth, COR_from_xth


def COR_2d_elastix(imagestack, thetas, rotation = 0):
    '''
    returns the cordinates of the COR in imagestack[0,:,:]
    rotation = aditional angle of the projection in xy plane in degrees
    '''
    dummy        = np.copy(imagestack)
    center       = 0.5*np.array(dummy[0].shape)
    thetas       = np.array(thetas) + rotation
        
    dummy, shift, thetas_found = ia.elastix_align(dummy, mode = 'rigid')
    
    xth = [(x,-thetas[i]) for i,(x,y) in enumerate(shift)]
    yth = [(y,-thetas[i]) for i,(x,y) in enumerate(shift)]
    xth = np.asarray(xth)
    yth = np.asarray(yth)

    print('center:   ', center)
    CORx      = center - np.array(COR_from_xth(xth))
    CORy      = center - np.array(COR_from_yth(yth))
    COR_found = 0.5*(CORx + CORy)
    print('COR before backprojection:') 
    print('fit to x-values:  ', CORx)
    print('fit to y-values:  ', CORy)
    print('average:          ', COR_found)
    
    ## this seems to be the right backprojection:
    COR       = 2*center - COR_found
    print('center of rotation found:')
    print(COR)

    d_thetas = [thetas[i] - found for i,found in enumerate(thetas_found)] 
    print('set angles   : ', thetas)
    print('found angles : ', thetas_found)
    print('error        : ', d_thetas)

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

    print('center:   ', center)
    CORx      = center - np.array(COR_from_xth(xth))
    CORy      = center - np.array(COR_from_yth(yth))
    COR_found = 0.5*(CORx + CORy)
    print('COR before backprojection:') 
    print('fit to x-values:  ', CORx)
    print('fit to y-values:  ', CORy)
    print('average:          ', COR_found)
    
    ## this seems to be the right backprojection:
    COR       = 2*center - COR_found
    print('center of rotation found:')
    print(COR)


    return COR

def COR_from_sideview(imagestack, thetas, mode = 'com',return_shift = False):
    '''
    gets the COR from an imagestack which is assumed to be the projection or side view of an object rotated by the angles given in thetas.
    returns relative coordinates of COR!
    mode can be :   'COM'       - center of mass
                    'elastix'   - elastix in 'translation' mode
                    'CC'        - cross correlation not implemeted!
    '''

    if mode.upper() == 'COM':
        linestack    = imagestack.sum(axis=1)
        COM          = nd.measurements.center_of_mass(linestack[0])
        dummy, shift = ia.centerofmass_align(linestack)
        shift = shift.reshape((shift.shape[0],))
        print(('shift ',shift))
        print(('shift.shape: ',shift.shape))
        #shift =  [dx for dx in shift]
        #print('shift ',shift)
        imageshift = np.asarray([[float(dx),0] for dx in shift])

    
        dummy = ia.shift_image(imagestack,imageshift)
        
        yth = [(x,thetas[i]) for i,x in enumerate(shift)]
        yth = np.asarray(yth)        
        print('COM in first frame:   ', COM)
        COR    = np.array(COR_from_yth(yth))
        print('found COR:            ', COR)
        COR[0] = COM - COR[0]
        print('center of rotation') 
        print('fit to x-values:  ', COR)

        
    elif  mode.upper() == 'ELASTIX':
        dummy = np.copy(imagestack)
        dummy, shift = ia.elastix_align(dummy, mode='translation')
        xth = [(x[1],thetas[i]) for i,x in enumerate(shift)]
        xth = np.asarray(xth)
        COR   = np.array(COR_from_xth(xth))
        COR[0] *= -1.0
        print('center of rotation') 
        print('fit to x-values:  ', COR)

    elif  mode.upper() == 'USERCLICK':
        dummy = np.copy(imagestack)
        dummy, shift = ia.userclick_align(dummy)
        xth = [(x[1],thetas[i]) for i,x in enumerate(shift)]
        xth = np.asarray(xth)
        COR   = np.array(COR_from_xth(xth))
        COR[0] *= -1.0
        print('center of rotation') 
        print('fit to x-values:  ', COR)

        
    elif mode.upper() == 'CC':
        dummy = np.copy(imagestack)
        dummy, shift = ia.crosscorrelation_align_1d(dummy, mode='translation')
        xth = [(x[1],thetas[i]) for i,x in enumerate(shift)]
        xth = np.asarray(xth)
        COR   = np.array(COR_from_xth(xth))
        COR[0] *= -1.0
        print('center of rotation') 
        print('fit to x-values:  ', COR)
    else:
        raise NotImplementedError( mode,' is not an implemented COR_from_sideview mode')

    if return_shift:
        return dummy, COR, shift    
    
    return dummy, COR

def COR_1d_COM(linestack, thetas, rotation=0, relative = False):
    '''
    return the coordinates of the COR 
    lines stacked in linestack.shape[0]
    lines linestack are projections along 'x', i.e. linestack.shape[1] is x for projection = 0
    rotation = angle of the projection in xy plane in degrees
    '''
    dummy        = np.copy(linestack)
    dummy, shift = ia.centerofmass_align(dummy)
    
    thetas = np.array(thetas) + rotation
    
    print('COM in first frame:   ', COM)
    COR    = np.array(COR_from_yth(yth))
    print('found COR:            ', COR)
    if relative:
        COR[0] *= -1.0
    else:
        COR[0] = COM[0] - COR[0]
    print('center of rotation') 
    print('fit to x-values:  ', COR)
    
    return dummy, COR

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

    print('COM:   ', COM)
    CORx   = COM - np.array(COR_from_xth(xth))
    CORy   = COM - np.array(COR_from_yth(yth))
    COR    = 0.5*(CORx + CORy)
    print('center of rotation') 
    print('fit to x-values:  ', CORx)
    print('fit to y-values:  ', CORy)
    print('average:          ', COR)
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


def COR_from_topview(imagestack, thetas, mode = 'COM', align = True):
    '''
    if align = True, aligns image_stack in place
    properies['modes'] available:
    'COM', 'CC', 'elastix'
    '''
    if mode.upper() == 'COM':
        print('calculating COR using the center of mass')
        COR = COR_2d_COM(imagestack,thetas)
    elif mode.upper() == '2D_CROSSCORRELATION':
        print('calculating COR using cross correlation')
        COR = COR_2d_crosscorrelation(imagestack,thetas)
    elif mode.upper() == 'ELASTIX':
        print('calculating COR using elastix')
        COR = COR_2d_elastix(imagestack,thetas)
    else:
        print('available modes are: \nCOM, CC, and soon elatix')
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

    aligned, COR_found = COR_from_topview(rotated, thetas)

    fig3, ax3 = plt.subplots(1)
    ax3.matshow(aligned.sum(0))
    ax3.plot(COR_found[1],COR_found[0],'go')    
    ax3.plot(COR[1],COR[0],'rx')
    ax3.set_title('summed, realigned dataset, set COR = red, found COR green')
    
    print('given COR: ', COR)
    print('found COR: ', COR_found)
    
    return True
