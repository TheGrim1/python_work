from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.optimize import leastsq
import scipy.ndimage as nd
import warnings
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
# credit to spectrocrunch (Wout De Nolf (wout.de_nolf@esrf.eu)) for the fitting template

from simplecalc.fitting import do_sin_fit
from simplecalc.fitting import do_sin360period_fit


def circle_func(p,t):
    '''
    x0 = p[0]
    y0 = p[1]
    phi0 = p[2]
    r = p[3]
    phi=t
    returns [x,y]
    '''
    x0 = p[0]
    y0 = p[1]
    phi0 = p[2]/180.*np.pi
    r = p[3]
    phi=np.asarray(t)/180.*np.pi

    return[(x0+r*np.cos(phi-phi0)),(y0+r*np.sin(phi-phi0))]

def circle_errorf(p,t):
    x0 = p[0]
    y0 = p[1]
    phi0 = p[2]
    r = p[3]
    phi=t[0]
    x=t[1]
    y=t[2]

    tx,ty=circle_func(p,t[0])
    
    return np.sqrt((x-tx)**2+(y-ty)**2)

def guess_circle(phi,x,y, verbose=False):

    datax = np.asarray([phi,x]).T
    datay = np.asarray([phi,y]).T

    betax = do_sin360period_fit(datax,verbose=False)
    betay = do_sin360period_fit(datay,verbose=False)
    print(betax[1]-90, betay[1])
    guess = betax[2],betay[2],(betax[1]+90.+betay[1])/2.,(betax[0]+betay[0])/2.
    
    return(guess)

def do_fit_circle_in_2d_data(phi,x,y,verbose=False):
    '''
    returns
    p = x0,y0,phi0,r
    for minimizing the distance from the circle to x and y datapoints (least square)
    phi,x,y must rotate math positive!
    i.e. min of
    (y-y0-r*sin(phi-phi0))**+(x-x0-r*cos(phi-phi0))**
    a la simplecalc.fitting2d
    '''
    
    guess = guess_circle(phi,x,y,verbose=verbose)
    data = np.asarray([phi,x,y])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, success = leastsq(circle_errorf, guess, args=(data))
        success = success>0 and success<5

    # print(circle_errorf(guess,data).sum())
    # print(circle_errorf(p,data).sum())
    
    if verbose:
        fig, ax = plt.subplots()
        ax.plot(y,x,'bo')
        res_x = x-circle_func(p,phi)[0]
        res_y = y-circle_func(p,phi)[1]

        ax.plot(circle_func(guess,phi)[1],circle_func(guess,phi)[0],'g--')

        plt.show()

        fig, ax = plt.subplots()
        ax.plot(res_x,'rx')
        ax.plot(res_y,'bx')
        ax.set_title('residuals')
        plt.show()
        fig, ax = plt.subplots()
        
        ax.plot(res_y,res_x,'r--')
        ax.set_title('residuals')
        plt.show()
    
    return p, success



def gauss2d_func(p,x,y,force_positive=False):
    x0,y0,sx,sy,rho,A = tuple(p)

    num = (x-x0)**2/sx**2 - 2*rho/(sx*sy)*(x-x0)*(y-y0) + (y-y0)**2/sy**2
    denom = 2*(1-rho**2)
    if force_positive:
        return np.abs(A/(2*np.pi*sx*sy*np.sqrt(1-rho**2))*np.exp(-num/denom))
    else:
        return A/(2*np.pi*sx*sy*np.sqrt(1-rho**2))*np.exp(-num/denom)

def gauss2d_errorf(p,data,x,y,force_positive=False):
    return np.ravel(gauss2d_func(p,x,y,force_positive)-data)

def guess_gauss2d(data,x,y):
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

def do_gauss2d_fit(data,x=None,y=None,force_positive=False):

    if type(x)==type(None):
        x,y=np.meshgrid(range(data.shape[1]),range(data.shape[0]))

    guess = guess_gauss2d(data,x,y)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, success = leastsq(gauss2d_errorf, guess, args=(data,x,y,force_positive))
        success = success>0 and success<5
        
    return right_sign(p), success

def zip_p(p):
    return zip(p[0::6],p[1::6],p[2::6],p[3::6],p[4::6],p[5::6])

def right_sign(p):
    for i in [2,3]:
        p[i::6] = np.abs(p[i::6])
    return p

def multiple_gauss2d_func(p,x,y,force_positive=False):
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
        data += gauss2d_func(single_p,x,y,force_positive)
    return data

def multiple_gauss2d_errorf(p, data, x, y,force_positive=False):
    return np.ravel(multiple_gauss2d_func(p,x,y,force_positive)-data)

def do_multiple_gauss2d_fit(data, x=None, y=None,force_positive=False):

    if type(x)==type(None):
        x,y=np.meshgrid(range(data.shape[1]),range(data.shape[0]))
    
    index_guess = localmax_peak_index_guess(data,no_peaks=10)
    
    # print(index_guess)
    if len(index_guess)!=0:
        guess = index_to_gauss2d_parameter_guess(index_guess, data, x, y)

        # print(guess)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, success = leastsq(multiple_gauss2d_errorf, guess, args=(data,x,y,force_positive))
            success = success>0 and success<5
        return right_sign(p)
    else:
        return np.zeros(6)

def localmax_peak_index_guess(data, min_distance=5, no_peaks = 5):
    '''
    filter with 'gauss' of width peak_width
    peaks < 0.1 are thrown away
    return multi_gauss2d_func parameter guess
    
    '''
    w_data = np.copy(data)
    w_data = nd.filters.gaussian_filter(w_data, sigma = min_distance, truncate=2.0)
    np.where(w_data<0.1,0,w_data)

    return peak_local_max(w_data, min_distance=min_distance, threshold_rel=0.0001,threshold_abs=3,num_peaks=no_peaks)

def index_to_gauss2d_parameter_guess(index_list, data, x, y):
    p = []
    for index in index_list:
        # print(index)
        max_val = data[index[0],index[1]]
        # x0
        p.append(x[index[0],index[1]])
        # y0
        p.append(y[index[0],index[1]])
        sx = np.abs(x[index[0]-5,index[1]] - x[index[0]+5,index[1]])
        # print('sx {}'.format(sx))
        p.append(sx)
        sy = np.abs(y[index[0],index[1]-5] - y[index[0],index[1]+5])
        # print(sy)
        p.append(sy)
        rho = 0.
        p.append(rho)
        A =data[index[0],index[1]]*2*np.pi*sx*sy*np.sqrt(1-rho**2)
        p.append(A)
        # print(A)

    return p
