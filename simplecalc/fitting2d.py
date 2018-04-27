from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.optimize
import scipy.ndimage as nd
import warnings
from skimage.feature import peak_local_max
# credit to spectrocrunch (Wout De Nolf (wout.de_nolf@esrf.eu))

def gauss2d_func(x,y,p):
    x0,y0,sx,sy,rho,A = tuple(p)
    num = (x-x0)**2/sx**2 - 2*rho/(sx*sy)*(x-x0)*(y-y0) + (y-y0)**2/sy**2
    denom = 2*(1-rho**2)
    return A/(2*np.pi*sx*sy*np.sqrt(1-rho**2))*np.exp(-num/denom)

def gauss2d_errorf(p,x,y,data):
    return np.ravel(gauss2d(x,y,p)-data)

def guess_gauss2d(x,y,data):
    y0i,x0i = np.unravel_index(np.argmax(data),data.shape)
    y0 = y[y0i,0]
    x0 = x[0,x0i]

    print(y0,x0)
    
    xv = x[y0i,:]-x0
    yv = data[y0i,:]
    sx = np.sqrt(abs(xv**2*yv).sum()/yv.sum())
    xv = y[:,x0i]-y0
    yv = data[:,x0i]
    sy = np.sqrt(abs(xv**2*yv).sum()/yv.sum())
    rho = 0.

    A = data[y0i,x0i]*2*np.pi*sx*sy*np.sqrt(1-rho**2)

    return np.array([x0,y0,sx,sy,rho,A],dtype=np.float32)

def do_gauss2d_fit(x,y,data):
    guess = guess_gauss2d(x,y,data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, success = scipy.optimize.leastsq(gauss2d_errorf, guess, args=(x,y,data))
        success = success>0 and success<5

    return p, success

def zip_p(p):
    return zip(p[0::6],p[1::6],p[2::6],p[3::6],p[4::6],p[5::6])

def multiple_gauss2d_func(x,y,p):
    '''
    p[0] = x0
    p[1] = y0
    p[2] = sx
    p[3] = sy
    p[4] = rho
    p[5] = A
    etc
    '''
    data = np.zeros(shape=x.shape,dtype=np.float32)
    for single_p in zip_p(p):
        data += gauss2d_func(x,y,single_p)
    return data

def multiple_gauss2d_errorf(p,x,y,data):
    return np.ravel(multiple_gauss2d_func(x,y,p)-data)

def do_multiple_gauss2d_fit(x,y,data):
    index_guess = localmax_peak_index_guess(data)
    
    guess = index_to_gauss2d_parameter_guess(index_guess, x, y, data)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, success = scipy.optimize.leastsq(multiple_gauss2d_errorf, guess, args=(x,y,data))
        success = success>0 and success<5

    return p

def localmax_peak_index_guess(data, min_distance=5):
    '''
    filter with 'gauss' of width peak_width
    peaks < 0.1 are thrown away
    return multi_gauss2d_func parameter guess
    
    '''
    w_data = np.copy(data)
    w_data = nd.filters.gaussian_filter(w_data, sigma = min_distance, truncate=2.0)
    np.where(w_data<0.1,0,w_data)

    return peak_local_max(w_data, min_distance=min_distance, threshold_rel=0.00001)

def index_to_gauss2d_parameter_guess(index_list, x, y, data):
    p = []
    for index in index_list:
        max_val = data[index[0],index[1]]
        # x0
        p.append(x[index[0],index[1]])
        # y0
        p.append(y[index[0],index[1]])
        sx = np.abs(x[index[0],index[1]-5] - x[index[0],index[1]+5])
        p.append(sx)
        sy =np.abs(y[index[0]-5,index[1]] - y[index[0]+5,index[1]])
        p.append(sy)
        rho = 0.
        p.append(rho)
        #A =
        p.append(data[index[0],index[1]]*2*np.pi*sx*sy*np.sqrt(1-rho**2))

    return p
