from __future__ import print_function
from __future__ import division

import numpy as np
from scipy.optimize import leastsq
from scipy.ndimage.filters import gaussian_filter, median_filter
import warnings

# credit to spectrocrunch (Wout De Nolf (wout.de_nolf@esrf.eu)) for the fitting template

from simplecalc.slicing import check_in_border

def gauss3d_func(p,xx,yy,zz,force_positive=False):
    '''
    http://mathworld.wolfram.com/TrivariateNormalDistribution.html
    '''
    y0,x0,z0,sy,sx,sz,sxy,sxz,syz,A = tuple(p)

    # rescaling axes (not in the reference):
    x1, x2, x3 = (xx-x0)/sx, (yy-y0)/sy, (zz-z0)/sz
    
    expnum1 = x1**2*(syz**2-1) + x2**2*(sxz**2-1) + x3**2*(sxy**2-1)
    expnum2 = 2*(x1*x2*(sxy - sxz*syz) + x1*x3*(sxz - sxy*syz) + x2*x3*(syz - sxy*sxz))
    expnum = expnum1+expnum2
    expdenom = 2*(sxy**2 + sxz**2 + syz**2 - 2*sxy*sxz*syz -1)

    # getting RuntimeWarnings for invalid numbers, maybe this helps
    Adenom = np.max([sx*sy*sz*(2*np.pi)**1.5*(1+2*sxy*sxz*syz - (sxy**2+sxz**2+syz**2))**0.5,1e-14])

    if force_positive:
        return np.abs(A/Adenom*np.exp(-expnum/expdenom))
    else:
        return A/Adenom*np.exp(-expnum/expdenom)

def gauss3d_errorf(p,data,xx,yy,zz,force_positive=False):
    return np.ravel(gauss3d_func(p,xx,yy,zz,force_positive)-data)

def guess_gauss3d(data,xx,yy,zz):
    y0i,x0i,z0i = np.unravel_index(np.argmax(data),data.shape)
    z0 = zz[y0i,x0i,z0i]
    y0 = yy[y0i,x0i,z0i]
    x0 = xx[y0i,x0i,z0i]

    # guess sigmas:
    xv = xx[y0i,:,z0i]-x0
    yv = np.abs(data[y0i,:,z0i])
    yv_sum = yv.sum()
    if yv_sum == 0:
        sx = 1
    else:
        sx = np.sqrt((xv**2*yv).sum()/yv_sum)
    
    xv = yy[:,x0i,z0i]-y0
    yv = np.abs(data[:,x0i,z0i])
    yv_sum = yv.sum()
    if yv_sum == 0:
        sy = 1
    else:
        sy = np.sqrt((xv**2*yv).sum()/yv_sum)

    
    xv = zz[y0i,x0i,:]-z0
    yv = np.abs(data[y0i,x0i,:])
    yv_sum = yv.sum()
    if yv_sum == 0:
        sz = 1
    else:
        sz = np.sqrt((xv**2*yv).sum()/yv_sum)

    
    Adenom = sx*sy*sz*(2*np.pi)
    
    A = data[y0i,x0i,z0i]* Adenom

    return np.array([y0,x0,z0,sy,sx,sz,0,0,0,A],dtype=np.float32)

def do_gauss3d_fit(data,xx=None,yy=None,zz=None, guess = None, force_positive=False):

    if type(xx)==type(None):
        xx,yy,zz=np.meshgrid(range(data.shape[1]),range(data.shape[0]),range(data.shape[2]))

    if type(guess) == type(None):
        guess = guess_gauss3d(data,xx,yy,zz)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, success = leastsq(gauss3d_errorf, guess, args=(data,xx,yy,zz,force_positive))
        success = success>0 and success<5
        
    return right_sign(p, force_positive), success

def zip_p(p):
    return zip(p[0::10],p[1::10],p[2::10],p[3::6],p[4::10],p[5::10],p[6::10],p[7::10],p[8::10],p[9::10])

def right_sign(p, force_positive=False):
    if force_positive:
        todo_indexes = [3,4,5,9]
    else:
        todo_indexes = [3,4,5]
    for i in todo_indexes:
        p[i::10] = np.abs(p[i::10])
    return p


def do_iterative_two_gauss3d_fit(data,
                                 xx=None,
                                 yy=None,
                                 zz=None,
                                 force_positive=False,
                                 diff_threshold=1,
                                 return_residual=False,
                                 max_iteration=10000,
                                 verbose=False):
    '''
    fit 2 gauss untill the sum of the movement of the two peaks (in xx/yy units) is less than diff_threshold
    does o medial filter befor fitting second peak - this greatly stabelizes fitting
    larger area peak will be returned first
    sets to np.nan if peak center is outside the data area as here the fitting is very instable!
    then (if possible) only one peak will be returned the second is set to nan on all parameters
    '''
    
    if type(xx)==type(None):
         
        xx,yy,zz=np.meshgrid(range(data.shape[1]),range(data.shape[0]),range(data.shape[2]))
        if verbose>2:
            print('generating coordinates')
            print('data.shape = ', data.shape)
            print('xx.shape = ', xx.shape)


    border = [[ymin, ymax], [xmin, xmax], [zmin, zmax], [A_min, A_max]] = [[yy.min(), yy.max()], [xx.min(), xx.max()], [zz.min(), zz.max()], [0 , 2*data.sum()]]
    diff = diff_threshold+1

    result1_list = [do_gauss3d_fit(data=data, xx=xx, yy=yy, zz=zz, force_positive=force_positive)[0]]
    gauss1 = gauss3d_func(result1_list[-1],xx,yy,zz)
    residual1 = data-gauss1
    result2_list = [do_gauss3d_fit(data=residual1, xx=xx, yy=yy, zz=zz, force_positive=force_positive)[0]]
    gauss2 = gauss3d_func(result2_list[-1],xx,yy,zz)
    residual2 = data - gauss2

    i=0
    while diff>diff_threshold:
        result1 = do_gauss3d_fit(data=residual2, xx=xx, yy=yy, zz=zz, guess=result1_list[-1], force_positive=force_positive)[0]
        gauss1 = gauss3d_func(result1,xx,yy,zz)

        residual1 = median_filter(data-gauss1, size=1)

        result2 = do_gauss3d_fit(data=residual1, xx=xx, yy=yy, zz=zz, guess=result2_list[-1], force_positive=force_positive)[0]
        gauss2 = gauss3d_func(result2,xx,yy,zz)
        residual2 = data - gauss2

        # check for failed fits
        b_check = [x for x in result1[:3]]
        b_check.append(result1[-1])
        if not check_in_border(b_check, border):
            if verbose>2:
                print('3D iteration {} hit border1'.format(i))
                print(zip(b_check,border))
            result1.fill(np.nan)

        b_check = [x for x in result2[:3]]
        b_check.append(result2[-1])
        if not check_in_border(b_check, border):
            if verbose>2:
                print('3D iteration {} hit border2'.format(i))
                print(zip(b_check, border))
            result2.fill(np.nan)
        
        if any(np.isnan(result1+result2)):
            result1 = do_gauss3d_fit(data=data, xx=xx, yy=yy, zz=zz, force_positive=force_positive)[0]
            if any(np.isnan(result1)):
                result1.fill(np.nan)
            result2.fill(np.nan)
            if return_residual:
                residual = (data - gauss3d_func(result1,xx,yy,zz)) / data.sum()
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
        diff2 = ((result2_list[-2][0]-result2_list[-1][0])**2 + (result2_list[-2][1]-result2_list[-1][1])**2 + (result2_list[-2][2]-result2_list[-1][2])**2)**0.5 
        diff1 = ((result1_list[-2][0]-result1_list[-1][0])**2 + (result1_list[-2][1]-result1_list[-1][1])**2 +  (result1_list[-2][2]-result1_list[-1][2])**2)**0.5
        diff = diff2 + diff1
        if verbose>2:
            print('iteration {}'.format(i))
            print('diff1 {:2.4f}, diff2 {:2.4f}'.format(diff1,diff2))
            print('diff {}'.format(diff))
        i += 1
        if i>max_iteration:
            break

    if verbose:
        print('found peaks {:2.2f}:{:2.2f} and {:2.2f}:{:2.2f}'.format(result1_list[-1][0],result1_list[-1][1],result2_list[-1][0],result2_list[-1][1]))
    if return_residual:
        residual = (data - gauss2 - gauss1) / data.sum()
        return np.asarray([result1,result2]),residual
    else:
        return np.asarray([result1_list[-1],result2_list[-1]])
