
from scipy.ndimage import laplace as ndlaplace
from scipy.ndimage import median_filter as ndmedian_filter
import numpy as np
import gauss_fitting

def focus_in_imagestack(imagestack, fit=True, verbose = False, filter_noise=True):
    '''
    returns the index of the image within imagestack that has the largest var() when convoluted with the laplacian -> it is most focussed
    if fit==True: fits a gaussian to the focal metric and returns its minimum
    '''

    [ndmedian_filter(imagestack[i],[5,5],output=imagestack[i]) for i in range(imagestack.shape[0])]
    focmetric_list = [(i,ndlaplace(image).var()) for (i,image) in enumerate(imagestack)]
    
    focmetric_array = np.asarray(focmetric_list)
    # subtract background
    
    if verbose:
        print('found focalmetric:')
        print focmetric_list
        
    if fit:
        focmetric_array[:,1] += -np.min(focmetric_array[:,1])
        fit_data = np.rollaxis(focmetric_array,-1)
        beta = gauss_fitting.do_gauss_plus_bkg_fit(fit_data, verbose=verbose)
        foc_index = beta[1]
        return foc_index, beta[2] 
    else:
        foc_index = np.argmax(focmetric_array[:,1])
        return foc_index


    
    


