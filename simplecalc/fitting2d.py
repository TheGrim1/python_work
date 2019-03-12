from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.optimize import leastsq
from scipy.ndimage.filters import gaussian_filter, median_filter
import warnings
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
# credit to spectrocrunch (Wout De Nolf (wout.de_nolf@esrf.eu)) for the fitting template

from simplecalc.fitting import do_sin_fit
from simplecalc.fitting import do_sin360period_fit

from simplecalc.slicing import check_in_border

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

def guess_circle(phi,xx,yy, verbose=False):

    datax = np.asarray([phi,xx]).T
    datay = np.asarray([phi,yy]).T

    betax = do_sin360period_fit(datax,verbose=False)
    betay = do_sin360period_fit(datay,verbose=False)
    print(betax[1]-90, betay[1])
    guess = betax[2],betay[2],(betax[1]+90.+betay[1])/2.,(betax[0]+betay[0])/2.
    
    return(guess)

def do_fit_circle_in_2d_data(phi,xx,yy,verbose=False):
    '''
    returns
    p = x0,y0,phi0,r
    for minimizing the distance from the circle to xx and yy datapoints (least square)
    phi,xx,yy must rotate math positive!
    i.e. min of
    (yy-y0-r*sin(phi-phi0))**+(xx-x0-r*cos(phi-phi0))**
    a la simplecalc.fitting2d
    '''
    
    guess = guess_circle(phi,xx,yy,verbose=verbose)
    data = np.asarray([phi,xx,yy])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, success = leastsq(circle_errorf, guess, args=(data))
        success = success>0 and success<5

    # print(circle_errorf(guess,data).sum())
    # print(circle_errorf(p,data).sum())
    
    if verbose:
        fig, ax = plt.subplots()
        ax.plot(yy,xx,'bo')
        res_x = xx-circle_func(p,phi)[0]
        res_y = yy-circle_func(p,phi)[1]

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



def gauss2d_func(p,xx,yy,force_positive=False):
    y0,x0,sy,sx,rho,A = tuple(p)

    num = (xx-x0)**2/sx**2 - 2*rho/(sx*sy)*(xx-x0)*(yy-y0) + (yy-y0)**2/sy**2
    denom = 2*(1-rho**2)

    Adenom = np.max([2*np.pi*sx*sy*np.sqrt(1-rho**2),1e-14])
    
    if force_positive:
        return np.abs(A)/Adenom*np.exp(-num/denom)
    else:
        return A/Adenom*np.exp(-num/denom)

def gauss2d_errorf(p,data,xx,yy,force_positive=False):
    return np.ravel(gauss2d_func(p,xx,yy,force_positive)-data)

def guess_gauss2d(data,xx,yy):
    y0i,x0i = np.unravel_index(np.argmax(data),data.shape)
    y0 = yy[y0i,0]
    x0 = xx[0,x0i]
   
    xv = xx[y0i,:]-x0
    yv = np.abs(data[y0i,:])
    yv_sum = yv.sum()
    if yv_sum == 0:
        sx = 1
    else:
        sx = np.sqrt((xv**2*yv).sum()/yv_sum)
    
    xv = yy[:,x0i]-y0
    yv = np.abs(data[:,x0i])
    yv_sum = yv.sum()
    if yv_sum == 0:
        sy = 1
    else:
        sy = np.sqrt((xv**2*yv).sum()/yv_sum)
    
    rho = 0.

    A = data[y0i,x0i]*2*np.pi*sx*sy*np.sqrt(1-rho**2)

    return np.array([y0,x0,sy,sx,rho,A],dtype=np.float32)

def do_gauss2d_fit(data,xx=None,yy=None, guess=None, force_positive=False):

    if type(xx)==type(None):
        xx,yy=np.meshgrid(range(data.shape[1]),range(data.shape[0]))

    if type(guess) == type(None):
        guess = guess_gauss2d(data,xx,yy)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, success = leastsq(gauss2d_errorf, guess, args=(data,xx,yy,force_positive))
        success = success>0 and success<5
            
    return right_sign(p, force_positive), success

def zip_p(p):
    return zip(p[0::6],p[1::6],p[2::6],p[3::6],p[4::6],p[5::6])

def right_sign(p, force_positive=False):
    if force_positive:
        todo_indexes = [2,3,6]
    else:
        todo_indexes = [2,3]
    for i in todo_indexes:
        p[i::6] = np.abs(p[i::6])
    return p

def do_iterative_two_gauss2d_fit(data,
                                 xx=None,
                                 yy=None,
                                 force_positive=False, 
                                 diff_threshold=1,
                                 max_iteration=10000,
                                 return_residual=False,
                                 verbose=False):
    '''
    fit 2 gauss untill the sum of the movement of the two peaks (in xx/yy units) is less than diff_threshold
    '''
    
    if type(xx)==type(None):
        if verbose:
            print('generating coordinates')
        xx,yy=np.meshgrid(range(data.shape[1]),range(data.shape[0]))
        
    diff = diff_threshold+1
    border = [[ymin, ymax], [xmin, xmax], [A_min, A_max]] = [[yy.min(), yy.max()], [xx.min(), xx.max()], [0.0, 2*data.sum()]]
    
    result1_list = [do_gauss2d_fit(data=data, xx=xx, yy=yy, force_positive=force_positive)[0]]
    gauss1 = gauss2d_func(result1_list[-1],xx,yy)
    residual1 = data-gauss1

    result2_list = [do_gauss2d_fit(data=residual1, xx=xx, yy=yy, force_positive=force_positive)[0]]
    gauss2 = gauss2d_func(result2_list[-1],xx,yy)
    residual2 = data - gauss2

    i=0
    while diff>0.01:
        result1 = do_gauss2d_fit(data=residual2, xx=xx, yy=yy, guess=result1_list[-1], force_positive=force_positive)[0]
        gauss1 = gauss2d_func(result1_list[-1],xx,yy)
        
        residual1 = median_filter(data-gauss1, size=1)

        result2 = do_gauss2d_fit(data=residual1, xx=xx, yy=yy, guess=result2_list[-1], force_positive=force_positive)[0]
        gauss2 = gauss2d_func(result2_list[-1],xx,yy)
        residual2 = data - gauss2

        # check for failed fits
        b_check = [x for x in result1[:2]]
        b_check.append(result1[-1])
        if not check_in_border(b_check, border):
            if verbose>2:
                print('2D iteration {} hit border1'.format(i))
                print(zip(b_check, border))
            result1.fill(np.nan)

        b_check = [x for x in result2[:2]]
        b_check.append(result2[-1])
        if not check_in_border(b_check, border):
            if verbose>2:
                print('2D iteration {} hit border2'.format(i))
                print(zip(b_check, border))
            result2.fill(np.nan)
        
        if any(np.isnan(result1+result2)):
            if verbose>2:
                print('found a nan in result1 OR result2')
            result1 = do_gauss2d_fit(data=data, xx=xx, yy=yy, force_positive=force_positive)[0]
            if any(np.isnan(result1)):
                if verbose>2:
                    print('immediate nan on fist fit that what you ll get')
                result1.fill(np.nan)
            result2.fill(np.nan)
            if return_residual:
                residual = (data - gauss2d_func(result1,xx,yy)) / data.sum()
                return np.asarray([result1,result2]),residual
            else:
                return np.asarray([result1,result2])

        # sort peaks accoring to Area
        if result2[-1] > result1[-1]:
            residual2 = residual1
            result1_list.append(result2)
            result2_list.append(result1)
        else:
            result1_list.append(result1)
            result2_list.append(result2)

         # check distance the peaks moved on this iteration
        diff2 = ((result2_list[-2][0]-result2_list[-1][0])**2 + (result2_list[-2][1]-result2_list[-1][1])**2)**0.5 
        diff1 = ((result1_list[-2][0]-result1_list[-1][0])**2 + (result1_list[-2][1]-result1_list[-1][1])**2)**0.5
        diff = diff2 + diff1
        i += 1
        if i>max_iteration:
            break
    if return_residual:
        residual = (data - gauss2 - gauss1) / data.sum()
        return np.asarray([result1,result2]),residual
    else:
        return np.asarray([result1,result2])  


        

def multiple_gauss2d_func(p,xx,yy,force_positive=False):
    '''
    p[0] = y0
    p[1] = x0
    p[2] = sy
    p[3] = sx
    p[4] = rho
    p[5] = A
    etc
    '''
    data = np.zeros(shape=xx.shape,dtype=np.float32)
    for single_p in zip_p(p):
        data += gauss2d_func(single_p,xx,yy,force_positive)
    return data

def multiple_gauss2d_errorf(p, data, xx, yy,force_positive=False):
    return np.ravel(multiple_gauss2d_func(p,xx,yy,force_positive)-data)

def do_multiple_gauss2d_fit(data, xx=None, yy=None,force_positive=False):

    if type(xx)==type(None):
        xx,yy=np.meshgrid(range(data.shape[1]),range(data.shape[0]))
    
    index_guess = localmax_peak_index_guess(data,no_peaks=10)
    
    # print(index_guess)
    if len(index_guess)!=0:
        guess = index_to_gauss2d_parameter_guess(index_guess, data, xx, yy)

        # print(guess)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, success = leastsq(multiple_gauss2d_errorf, guess, args=(data,xx,yy,force_positive))
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
    w_data = gaussian_filter(w_data, sigma = min_distance, truncate=2.0)
    np.where(w_data<0.1,0,w_data)

    return peak_local_max(w_data, min_distance=min_distance, threshold_rel=0.0001,threshold_abs=3,num_peaks=no_peaks)

def index_to_gauss2d_parameter_guess(index_list, data, xx, yy):
    p = []
    for index in index_list:
        # print(index)
        max_val = data[index[0],index[1]]
        # x0
        p.append(xx[index[0],index[1]])
        # y0
        p.append(yy[index[0],index[1]])
        sx = np.abs(xx[index[0]-5,index[1]] - xx[index[0]+5,index[1]])
        # print('sx {}'.format(sx))
        p.append(sx)
        sy = np.abs(yy[index[0],index[1]-5] - yy[index[0],index[1]+5])
        # print(sy)
        p.append(sy)
        rho = 0.
        p.append(rho)
        A =data[index[0],index[1]]*2*np.pi*sx*sy*np.sqrt(1-rho**2)
        p.append(A)
        # print(A)

    return p
