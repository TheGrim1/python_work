from __future__ import print_function
from __future__ import division

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.ndimage as nd
from scipy.ndimage.filters import median_filter as med_fil

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))


from simplecalc import fitting



def clean_outliers(data,outlier_factor, median_radius):
    '''
    will replace all datapoints that are > data.mean * (1 + outlier_factor) or < data.mean * (1 - outlier_factor)
    with the median in radius (scipy.ndimage.filters.median_filter)
    '''
    result = np.copy(data)
    med_filterd = med_fil(data, median_radius)
    data_mean = np.mean(data)
    result = np.where(data>data_mean * (1 + outlier_factor), med_filterd, data)
    result = np.where(result<data_mean * (1 - outlier_factor), med_filterd, result)
    return result


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def combine_datasets(datadict):
    '''
    combines a dict of datasets into one array on a common x axis.
    the x axis spans from the min to max off all datasets and is 2 x the longest dataset in length.
    each yn is created by interpolating to dataset(xn)
    '''
    xmin    = None
    xmax    = None
    xlen    = None
    
    dataheader = list(datadict.keys())
    dataheader.sort()
    for fname, dataset in list(datadict.items()):
        if xmin == None:
            xmin = np.min(dataset[:,0])
            xmax = np.max(dataset[:,0])
            xlen = dataset.shape[0]
        else:
            xmin = min(np.min(dataset[:,0]),xmin)
            xmax = max(np.max(dataset[:,0]),xmax)
            xlen = max(dataset.shape[0],xlen)

    dataheader.insert(0,'common x')
    xaxis = np.atleast_1d(np.arange(xmin,xmax, ((float(xmax-xmin))/(2*xlen))))
    fulldata = np.zeros(shape = (len(xaxis), len(dataheader)))
#    print xaxis.shape
#    print fulldata.shape
    fulldata[:,0] = xaxis

    for i, fname in enumerate(dataheader[1::]):
        dataset = datadict[fname]
        fulldata[:,i+1] = np.interp(xaxis, dataset[:,0], dataset[:,1])
        
    return fulldata, dataheader

def normalize_xanes(data, e0, preedge, postedge, fitorder = 1, edgemode = 'fit', verbose = False):
    '''
    data[0] = energy, (must be sorted!)
    data[1] = mu(E)
   

    data = data - <fitorder 1 or 2> polynomial fit on where data[0][:]  >= preedge[0] and <= preedge[1]
    data = data / <fitorder> polynomial fit on where data[0][:]  >= postedge[0] and <= postedge[1]
    returns:
    data as above
    edge where data = 0.5 for the first time
    step = post_edge(e0) - pre_edge(e0)
    '''

    predatafail = False
    prestart = np.searchsorted(data[:,0], preedge[0], 'right')
    preend   = np.searchsorted(data[:,0], preedge[1], 'left')
    predata  = data[np.arange(prestart, preend),:]
    predata  = predata[np.where(predata[:,1]!=0),:][0]
    if len(predata[:,0]) < 4:
        predatafail = True
        if verbose:
            print('pre-edge fitting failed')

    postdatafail = False
    poststart = np.searchsorted(data[:,0], postedge[0], 'right')
    postend   = np.searchsorted(data[:,0], postedge[1], 'left')
    postdata  = data[np.arange(poststart, postend),:]
    postdata  = postdata[np.where(postdata[:,1]!=0),:][0]
    if len(postdata[:,0]) < 4:
        postdatafail = True
        if verbose:
            print('post-edge fitting failed')

    if verbose:
        print('found predata.shape')
        print(predata.shape)
        print('found postdata.shape')
        print(postdata.shape)
        
    ### define the pre_edge and post_edge functions using a fit
    if verbose > 3:
        verbose2 = True
    else:
        verbose2 = False

    if fitorder == 1:
        prefitpar = fitting.do_linear_fit(predata,verbose = verbose2)
        def pre_edge(x):
            return fitting.linear_func(prefitpar,x)
        postfitpar = fitting.do_linear_fit(postdata,verbose = verbose2)
        def post_edge(x):
            return fitting.linear_func(postfitpar,x)
    elif fitorder == 2:
        prefitpar = fitting.do_quadratic_fit(predata,verbose = verbose2)
        def pre_edge(x):
            return fitting.quadratic_func(prefitpar,x)
        postfitpar = fitting.do_quadratic_fit(postdata,verbose = verbose2)
        def post_edge(x):
            return fitting.quadratic_func(postfitpar,x)
    else:
        print('fitorder = %s is not implemented!' % fitorder)
        sys.exit()

    ### doing the normalization

    normdata = np.zeros(shape=data.shape)
    normdata[:,0] = data[:,0]
    if not predatafail:
        normdata[:,1] = data[:,1] - pre_edge(data[:,0])
    else:
        normdata[:,1] = data[:,1]
    
    ### if the fit functions cross, don't devide
    diff = (post_edge(data[:,0]) - pre_edge(data[:,0]))
    if np.any(np.where(diff <=0)):
        postdatafail = True
        if verbose:
            print('anormal normalization function, not normalizing')

    if not postdatafail:
        normdata[:,1] = normdata[:,1]/ diff
    else:
        normdata[:,1] = normdata[:,1]

    ### find step- up and edge
    if predatafail or postdatafail:
        step = 0
        edge = 0
    else:
        step = post_edge(e0) - pre_edge(e0)

        ### edge fit or find
        try:
            edgeindex = np.where(normdata[:,1] >=0.5)[0][0]
            if edgeindex <= 5:
                edge = 0
            else:
                edge      = normdata[edgeindex, 0]
        except IndexError:
            edge = 0

        ### fit cubic function at the edge +- 4 points  and look where it is 0.5
        
        if edgemode == 'fit' and not edge == 0:
            fitrange   = normdata[np.arange(edgeindex - 4, edgeindex +4),:]
            if verbose:
                print('fitting edge at index {} around energy {}:'.format(str(edgeindex),str(edge)))
                print(fitrange)

            edgefitpar = fitting.do_cubic_fit(fitrange, verbose = verbose2)
            def at_edge(x):
                return fitting.cubic_func(edgefitpar, x)
            edge = find_value(at_edge, 0.5, edge)

            if verbose:
                print('found edge at {}'.format(str(edge))) 

        
    ### optional plot:
        
    if verbose and not (postdatafail or predatafail):
        plt.clf()
        ax1 = plt.gca()
        ax1.plot(data[:,0],data[:,1],color = 'blue', linewidth = 2)
        ax1.plot(data[:,0],pre_edge(data[:,0]),color = 'red', linewidth = 2)
        ax1.plot(data[:,0],post_edge(data[:,0]),color = 'green', linewidth = 2)
        ax1.vlines(preedge[1],min(data[:,1]),max(data[:,1]),color = 'red', linewidth = 2)
        ax1.vlines(postedge,min(data[:,1]),max(data[:,1]),color = 'green', linewidth = 2)
        ax1.vlines([e0],min(data[:,1]),max(data[:,1]),color = 'blue', linewidth = 2)
        ax1.set_title('results of fitting')
        ax1.set_ylabel('signal [arb.]')
        ax1.set_xlabel('energy [eV]')
        plt.tight_layout()
        plt.show()
        ax1 = plt.gca()
        normdata[:,1] = nd.gaussian_filter1d(normdata[:,1],1)
        ax1.plot(normdata[:,0],normdata[:,1], linewidth = 2)
        if 'fitrange' in locals():
            ax1.vlines([edge],min(normdata[:,1]),max(normdata[:,1]),color = 'green')
            ax1.plot(fitrange[:,0],at_edge(fitrange[:,0]),color = 'red')
        ax1.set_ylabel('standardized signal [norm.]')
        ax1.set_xlabel('energy [keV]')
#        ax1.set_ylim([-0.2,2])
        ax1.set_title('standardized data, stepheight was {}'.format(str(step)))
        plt.tight_layout()
        plt.show()

    
        
    
    return (normdata, edge, step)
    
def normalize_self(data):
    '''
    data = data - np.min(data)
    data = data / np.max(data)
    '''

    data = data - np.min(data)
    data = data/ np.max(data)
    return data

def find_value(fun1, y, x0):
    '''
    find x0 where fun1(x0) = y
    '''
    return fsolve(lambda x : fun1(x) - y, x0)

def find_intersection(fun1, fun2, x0):
    '''
    2 functions x0 as starting guess
    '''
    return fsolve(lambda x : fun1(x) - fun2(x),x0)

def avg_array(basedata,newdata,n):
    
    basedata += newdata/ n
    
    return basedata

def get_fwhm(data):
    '''
    stack exchange http://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    '''

    if len(data.shape)==1:
        X = list(range(len(Y)))
        Y = data
    elif len(data.shape)==2:
        X = data[:,0]
        Y = data[:,1]
    else:
        print('invalid data shape!')

    half_max = np.max(Y)/ 2.
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))

    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
   
    return X[right_idx] - X[left_idx] #return the difference (full width)

def add_peaks(peak1, peak2):
    '''
    calculates the "sum" of two gaussian peaks.  
    ax = a(x+1) + ax ,
    muX +- sigmaX/2 overlaps with mu(x+1) +- sigma(x+1)/2
    i.e.:
    if np.absolute(muX - mu(x+1)) < [sigmaX + sigma(x+1)] /2:
    ax = a(x+1) + ax ,
    mux = (ax*mux + mu(x+1)*a(x+1))/(ax+a(x+1))  
    sigmaX = http://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians#16609
    '''
    a1   = peak1[0]
    mu1  = peak1[1]
    sig1 = peak1[2]
    a2   = peak2[0]
    mu2  = peak2[1]
    sig2 = peak2[2]
    
    a3   = a1+a2
    
    mu3  = (a1*mu1 + a2*mu2)/(a3)

    sig3 = (((a1*sig1**2 + a2*sig2**2) + (a1*mu1**2 + a2*mu2**2))/(a3)) - (((a1*mu1+a2*mu2)/a3))**2 

    peak3 = [a3,mu3,sig3]
    return peak3


def subtract_c_bkg(data, percentile = 20):
    background = np.empty(len(data[1,:]))

    background.fill(np.percentile(data[1,:], percentile))
    
    data[1,:]  += - background
    return data
    

def define_a_line_as_mask(exampledatashape, inclination=(-82.0/54), yintersect=100, width=4, verbose=False):
    '''
    not really on funbctional level, but worked once like this:
    '''
    ## masking the wire:
    mask = np.zeros(shape = exampledatashape)
    
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if line_maskfunc(x, y, inclination, yintersect, width):
                mask[y,x] = 1
            else:
                mask[y,x] = 0

    if verbose:
        plt.matshow(np.where(mask,1,0))                    
        plt.show()
    return mask

def line_maskfunc(x, y, inclination, yintersect, width):
    '''
    returns True if the point <(x,y)> is closer than <width> to the line defined by y' = inclination* x' + yintersect
    '''
    m = inclination
    c = yintersect
    if (np.abs(-m*x + y -c)/np.sqrt(m**2 +1)) <= 2:
        nearwire = True
    else:
        nearwire = False
    return nearwire

def define_a_circle_as_mask(exampledatashape, center_x, center_y, radius=4, verbose=False):
    '''
    not really on funbctional level, but worked once like this:
    '''
    ## masking the wire:
    mask = np.zeros(shape = exampledatashape)
    
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if circle_maskfunc(x, y, center_x, center_y, radius):
                mask[y,x] = 1
            else:
                mask[y,x] = 0

    if verbose:
        plt.matshow(np.where(mask,1,0))                    
        plt.show()
    return mask


def circle_maskfunc(x, y, cy, cx, radius):
    '''
    returns true if point xy in circle of radius around point (cx, cy)
    '''
    if ((cx-x)**2 + (cy - y)**2) < radius**2:
        return True
    else:
        return False

def get_hm_com(data):
    '''
    com of everything above the hm
    which is something like the spec cen
    '''
    min, max = data.min(), data.max()
    hm = (min+max)/2.
    top_data = np.where(data<hm,0,data)
    return nd.measurements.center_of_mass(top_data)
