import numpy as np
import scipy.ndimage as nd
import sys, os

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

from fileIO.plots.plot_array import plot_array
from fileIO.images.image_tools import imagefile_to_array
from fileIO.images.image_tools import array_to_imagefile

def rotate_series(array, thetas, COR = (785,1450),copy = True):
    '''
    maintains shape of array, but rotates array[n,:,:] by thetas[n] degrees
    '''
    COR         = np.array(COR)
    no_angles   = len(thetas)
    if copy:
        dummy = np.copy(array)
    else:
        dummy = array
    
    for i,th in enumerate(thetas):

        a         = th*np.pi/180.0
        transform = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]).dot(np.diag(([1,1])))
        offset    = COR-COR.dot(transform)
        
        dummy[i,:,:] = nd.interpolation.affine_transform(dummy[i,:,:],
                                                         transform.T,
                                                         order=2,
                                                         offset=offset,
                                                         mode='nearest',
                                                         cval=0.0)

    
    return dummy


    

def create_rotated_series(array, thetas, COR = (785,1450)):
    ''' returns massive array, no saving
    '''
    start_image = array
    COR         = np.array(COR)
    no_angles   = len(thetas)
    rotated     = np.zeros(shape = ([no_angles] + list(start_image.shape)))

    for i,th in enumerate(thetas):
        a         = th*np.pi/180.0
        transform = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]).dot(np.diag(([1,1])))
        offset    = COR-COR.dot(transform)
        
        rotated[i] = nd.interpolation.affine_transform(start_image,
                                                       transform.T,
                                                       order=2,
                                                       offset=offset,
                                                       mode='nearest',
                                                       cval=0.0)

    
    return rotated
