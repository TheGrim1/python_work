from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.optimize import leastsq, curve_fit
import math
import sys,os

from scipy.signal import find_peaks_cwt as find_peaks
from simplecalc.calc import subtract_c_bkg
from scipy.ndimage import gaussian_filter1d as gauss1d

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
from simplecalc.slicing import array_as_list


def peak_finder_1d(data,filterwidth = 1):
    '''
    my own attempt, replaced with
    from scipy.signal import find_peaks_cwt as find_peaks
    finds peaks in 2xn data, sorted according to data[peakpos]
    '''

    plt.plot(data[0],data[1],color = 'red')
    print(data.shape)
    print(data[0].shape)
    narrowfilter = gauss1d(data[1],sigma = filterwidth*0.8, mode = 'constant')
    print(narrowfilter.shape)
    plt.plot(data[0],narrowfilter,color = 'blue')

    widefilter   = gauss1d(data[1],sigma = filterwidth * 1.1,mode = 'constant')
    plt.plot(data[0],widefilter,color = 'darkblue')

    
    positive = narrowfilter - widefilter
    positive = np.where(positive >= 0, positive, -1)
    plt.plot(data[0],positive,color = 'green')
    
    peaks = []
    i     = 0
    flag  = False
    for point,i in enumerate(positive):
        if (flag and not point) or (point and not flag):
            peaks.append(i)
            plt.axvline(i,color='r') # edges of peakareas
     

    plt.show()
    return peakpos



def peak_guess(data, nopeaks = 2, verbose = False):
    '''
    trying to find good starting positions for the gauss fitting, returns nopeaks number of peaks
    print("p[0], a1: ", v[0])
    print("p[1], mu1: ", v[1])
    print("p[2], sigma1: ", v[2])
    print("p[3], a2: ", v[3])
    print("p[4], mu2: ", v[4])
    print("p[5], sigma2: ", v[5])
    '''

    width  = len(data[0])/10.0
    widths = []
    peaks  = []

    
    while len(peaks) < nopeaks:
        width *= 0.5
        widths.append(width)
        peak_indexes = find_peaks(data[1],widths = np.asarray(widths))

        if verbose:
            if len(peaks) >=1:
                print('with width = %s, guess found peaks at :'%width)
                print(data[0][peaks])

    peakheight = data[1][np.asarray(peak_indexes,dtype = np.int)]
    peakpos    = data[0][np.asarray(peak_indexes,dtype = np.int)]
    peaks      = np.asarray([peakheight,peakpos])
    peaks      = peaks[:,peaks[0,:].argsort()[::-1]]

    guess      = []
    sigmaguess = 5*np.absolute(data[0,0]-data[0,1])

    peakheight = data[1][np.asarray(peak_indexes,dtype = np.int)]
    peakpos    = data[0][np.asarray(peak_indexes,dtype = np.int)]
    

    if verbose:
        print(peakpos)
        print(peakheight)
    guess     = []

    for i in range(nopeaks):
        guess.extend([peaks[0,i]*sigmaguess,peaks[1,i],sigmaguess])    
#    guess.extend([peakheight[i],peakpos[i],width])

    if verbose and nopeaks == 2:
        print('guessing possitions with')
        print(("p[0], a1: ", guess[0]))
        print(("p[1], mu1: ", guess[1]))
        print(("p[2], sigma1: ", guess[2]))
        print(("p[3], a2: ", guess[3]))
        print(("p[4], mu2: ", guess[4]))
        print(("p[5], sigma2: ", guess[5]))
        
    return guess

def conservative_peak_guess(data,
                            nopeaks = 2,
                            verbose = False,
                            plot = False,
                            threshold = 10,
                            minwidth = 2,
                            maxwidth = None):
    '''
    trying to find good starting positions for the gauss fitting:
    print("p[0], a1: ", v[0])
    print("p[1], mu1: ", v[1])
    print("p[2], sigma1: ", v[2])
    print("p[3], a2: ", v[3])
    print("p[4], mu2: ", v[4])
    print("p[5], sigma2: ", v[5])
    '''
    if maxwidth == None:
        width  = len(data[0])/10.0
    else:
        width = maxwidth
    widths = []
    peaks  = []

    
    while len(peaks) < nopeaks and width > minwidth:
        widths.append(width)
        peaks = find_peaks(data[1],widths = np.asarray(widths))
        width *= 0.5

    if verbose:
        print('with minwidth = %s, guess found peaks at :'%width)
        print([peaks])

    if plot:
        plt.clf()
        plt.plot(list(range(len(data[1]))),data[1])
        for x in peaks:
            plt.axvline(x,color='r') # found peaks
            
    peakheight = data[1][peaks]
    peakpos    = data[0][peaks]
    peaks      = np.asarray([peakheight,peakpos])
    peaks      = peaks[:,peaks[0,:].argsort()[::-1]]


    guess      = []
    sigmaguess = 5*np.absolute(data[0,0]-data[0,1])
    
    for i in range(min(len(peaks[0]),nopeaks)):
        guess.extend([peaks[0,i]*sigmaguess,peaks[1,i],sigmaguess])

    if verbose:
        print('sorted into:')
        print(guess)
        
    if verbose and nopeaks == 2:
        print('guessing possitions with')
        print(("p[0], a1: ", guess[0]))
        print(("p[1], mu1: ", guess[1]))
        print(("p[2], sigma1: ", guess[2]))
        print(("p[3], a2: ", guess[3]))
        print(("p[4], mu2: ", guess[4]))
        print(("p[5], sigma2: ", guess[5]))
        
    return guess


def gauss_plus_bkg_func(p, t, force_positive=False): 
    '''
    p0 = a
    p1 = mu
    p2 = sigma
    p3 = c
    '''
    if force_positive:
        p[0]=np.absolute(p[0])
    return p[0]*(1.0/math.sqrt(2*math.pi*(p[2]**2)))*math.e**((-(t-p[1])**2/(2*p[2]**2))) + p[3]


def gauss_plus_bkg_residual(p, x, y, force_positive=False):
    return (gauss_plus_bkg_func(p,x,force_positive=force_positive) - y)

def do_gauss_plus_bkg_fit(data, verbose = False, force_positive=False):
    '''
    p0 = a
    p1 = mu
    p2 = sigma
    p3 = c
    '''
    constant_guess = np.min(data[1,:])
    sigma_guess = np.absolute(data[0,0]-data[0,1])
    a_guess = (np.max(data[1,:]) - constant_guess )*len(data[0,:])/ sigma_guess
    mu_guess    = data[0,:][np.argmax(data[1,:])]

    v0 = [a_guess, mu_guess, sigma_guess, constant_guess]
    
    if verbose == True:
        fig = plt.figure(figsize=(9, 9)) #make a plot
        ax1 = fig.add_subplot(111)
        ax1.plot(data[0],data[1],'gs') #spectrum
        ax1.plot(data[0],gauss_plus_bkg_func(v0,data[0]),'b') #fitted spectrum
        print('guess peaks at')
        print(("p[0], a1: ", v0[0]))
        print(("p[1], mu1: ", v0[1]))
        print(("p[2], sigma1: ", v0[2]))
        print(("p[3], c: ", v0[3]))
        plt.show()
        


    def optfunction(p,x,y):
        return gauss_plus_bkg_residual(p = p, x = x, y = y, force_positive=force_positive)
    out = leastsq(optfunction, v0, args=(data[0], data[1]), maxfev=100000, full_output=1) #Gauss Fit
    v = out[0] #fit parameters out
    covar = out[1] #covariance matrix output
    xxx = np.arange(min(data[0]),max(data[0]),data[0][1]-data[0][0])
    ccc = gauss_plus_bkg_func(v,xxx) # this will only work if the units are pixel and not wavelength

    if verbose == True:
        fig = plt.figure(figsize=(9, 9)) #make a plot
        ax1 = fig.add_subplot(111)
        ax1.plot(data[0],data[1],'gs') #spectrum
        ax1.plot(xxx,ccc,'b') #fitted spectrum
        print('found peak with')
        print(("p[0], a1: ", v[0]))
        print(("p[1], mu1: ", v[1]))
        print(("p[2], sigma1: ", v[2]))
        print(("p[3], c: ", v[3]))
        plt.show()
        
    return v


def gauss_func(p, t):
    '''
    p0 = a
    p1 = mu
    p2 = sigma
    '''
    return p[0]*(1.0/math.sqrt(2*math.pi*(p[2]**2)))*math.e**((-(t-p[1])**2/(2*p[2]**2)))

def two_gauss_func(p, t):
    return  gauss_func(p[0:3],t) + gauss_func(p[3:6],t)

def two_gauss_residual(p, x, y):
    return (two_gauss_func(p,x) - y)
#    e_gauss_fit = lambda p, x, y: (two_gauss_func(p,x) -y) #1d residual

def multi_gauss_func(p,t,nopeaks):
    function = np.zeros(shape = (len(t)))
    for i in range(nopeaks):
        function += gauss_func(p[list(range(i*3,(i+1)*3))],t)
    return function


def multi_gauss_residual(p,x,y,nopeaks):
    return multi_gauss_func(p,x,nopeaks = nopeaks) - y


def do_variable_gauss_fit(data, v0= None, plot = False, verbose = False, minwidth = 2, maxwidth = None):
    '''
    Fits with nopeaks = len(v0)/ number of gauss functions, if v==None, fits 3 peaks.
    Returns list(np.array(3,nopeaks)) containing sequence  of aX, muX, sigmaX at most nopeaks times.
    Also returns the residual of the least squares fit.
    v0 are the starting values for aX, muX, sigmaX etc.
    '''
    
    if v0 == None:
        v0      = conservative_peak_guess(data,
                                          nopeaks = 3,
                                          verbose = verbose,
                                          minwidth = minwidth,
                                          maxwidth = maxwidth)

    nopeaks = int(len(v0)/3)

    def optfunction(p,x,y):
        return multi_gauss_residual(p = p, x = x, y = y, nopeaks=nopeaks)
    out = leastsq(optfunction, v0[:], args=(data[0], data[1]), maxfev=100000, full_output=1) #Gauss Fit
    v = out[0] #fit parameters out
    covar = out[1] #covariance matrix output
    xxx = np.arange(min(data[0]),max(data[0]),data[0][1]-data[0][0])
    ccc = multi_gauss_func(v,xxx,nopeaks = nopeaks) # this will only work if the units are pixel and not wavelength

    residual = sum(optfunction(v,data[0],data[1]))
    if plot == True:
        fig = plt.figure(figsize=(9, 9)) #make a plot
        ax1 = fig.add_subplot(111)
        ax1.plot(data[0],data[1],'gs') #spectrum
        ax1.plot(xxx,ccc,'b') #fitted spectrum

    if verbose == True:
        print('found peaks at')
        l = 1
        for i, value in enumerate(v):
            if i%3 ==0:
                print("p[%s], a%s: %s" % (i,l,value))
            elif i%3 ==1:
                print("p[%s], mu%s: %s" % (i,l,value))
                if plot == True:
                    if value < max(xxx) and value > min(xxx):
                        plt.axvline(x = value,color = 'red')
            elif i%3 ==2:
                print("p[%s], sigma%s: %s" % (i,l,value))
                l +=1

    if plot == True:            
        plt.show()



            
    v = np.hstack([v,np.zeros(3*nopeaks -len(v))])
    return v.reshape((nopeaks,3)), residual


    
def do_multi_gauss_fit(data, nopeaks = 3, verbose = False):
    '''
    returns list containing sequence  of aX, muX, sigmaX nopeaks times
    '''
    v0 = peak_guess(data,nopeaks = nopeaks,verbose = verbose)
    def optfunction(p,x,y):
        return multi_gauss_residual(p = p, x = x, y = y, nopeaks=nopeaks)
    out = leastsq(optfunction, v0[:], args=(data[0], data[1]), maxfev=100000, full_output=1) #Gauss Fit
    v = out[0] #fit parameters out
    covar = out[1] #covariance matrix output
    xxx = np.arange(min(data[0]),max(data[0]),data[0][1]-data[0][0])
    ccc = multi_gauss_func(v,xxx,nopeaks = nopeaks) # this will only work if the units are pixel and not wavelength

    if verbose == True:
        fig = plt.figure(figsize=(9, 9)) #make a plot
        ax1 = fig.add_subplot(111)
        ax1.plot(data[0],data[1],'gs') #spectrum
        ax1.plot(xxx,ccc,'b') #fitted spectrum
        print('found peaks at')
        l = 1
        for i, value in enumerate(v):
            if i%3 ==0:
                print("p[%s], a%s: %s" % (i,l,value))
            elif i%3 ==1:
                print("p[%s], mu%s: %s" % (i,l,value))
                if value < max(xxx) and value > min(xxx):
                    plt.axvline(x = value,color = 'red')
            elif i%3 ==2:
                print("p[%s], sigma%s: %s" % (i,l,value))
                l +=1
        plt.show()

    return v.reshape((3,nopeaks))

def do_two_gauss_fit(data, verbose = False):
    '''
    old ->
    dev version before multi_gauss_fit
    '''
    v0 = peak_guess(data,nopeaks = 2,verbose = verbose)
    out = leastsq(two_gauss_residual, v0[:], args=(data[0], data[1]), maxfev=100000, full_output=1) #Gauss Fit
    v = out[0] #fit parameters out
    covar = out[1] #covariance matrix output
    xxx = np.arange(min(data[0]),max(data[0]),data[0][1]-data[0][0])
    ccc = two_gauss_func(v,xxx) # this will only work if the units are pixel and not wavelength

    if verbose == True:
        fig = plt.figure(figsize=(9, 9)) #make a plot
        ax1 = fig.add_subplot(111)
        ax1.plot(data[0],data[1],'gs') #spectrum
        ax1.plot(xxx,ccc,'b') #fitted spectrum
        plt.show()
        print('found peaks at')
        print(("p[0], a1: ", v[0]))
        print(("p[1], mu1: ", v[1]))
        print(("p[2], sigma1: ", v[2]))
        print(("p[3], a2: ", v[3]))
        print(("p[4], mu2: ", v[4]))
        print(("p[5], sigma2: ", v[5]))

    return v.reshape((3,2))



def do_variable_gaussbkg_pipeline(data,
                                  nopeaks =3,
                                  plot = False,
                                  verbose = False,
                                  threshold = 10,
                                  minwidth = 20,
                                  maxwidth = None):

    initial_v0 = conservative_peak_guess(data=data,
                                         nopeaks=nopeaks,
                                         verbose=verbose,
                                         plot=plot,
                                         threshold=threshold,
                                         minwidth=minwidth,
                                         maxwidth=maxwidth)

    v0 = []
    
    for i in range(int(len(initial_v0)/3)):
        peakindex = np.searchsorted(data[0], initial_v0[i*3+1])
        peakheight = data[1][peakindex]
        if peakheight > threshold:
            v0.extend([initial_v0[i*3],initial_v0[i*3+1],initial_v0[i*3+2]])

    if verbose>2:
        print('initial peaks guessed was:')
        print(initial_v0)
        print('after thresholding at %s' % threshold)
        print(v0)


    data = subtract_c_bkg(data, percentile = 20)
    
    if not len(v0) == 0:
        peaks, residual = do_variable_gauss_fit(data=data,
                                                v0=v0,
                                                plot=plot,
                                                verbose=verbose)

        peaks = np.hstack([peaks.flat,np.zeros(3*nopeaks -len(peaks.flat))])
        peaks = peaks.reshape((nopeaks,3))

    else:
        peaks = np.zeros(shape = (nopeaks,3))
        residual = sum(data[1])
        
        if verbose > 2:
            print('No peaks found in this frame')
    
    
    return peaks, residual



def fit_2d_gauss(array):
    '''
    returns params, residual
    params as [amp, x0, y0, a, b, c] see .gauss2d
    '''
    print('array.shape: ', array.shape)
    xyz   = array_as_list(array)
    xy    = xyz[0:2]
    z     = xyz[2]
    i     = z.argmax()

    guess = [1, xy[0][i], xy[1][i], 1, 1, 1]
    params_found, uncert_cov = curve_fit(gauss2d, xy, z, p0=guess)

    zpred = gauss2d(xy, *params_found)
    
    residual = np.sqrt(np.mean((z - zpred)**2))
    return params_found, residual

def gauss2d(xy, amp, x0, y0, a, b, c):
    '''
    # from https://stackoverflow.com/questions/27539933/2d-gaussian-fit-for-intensities-at-certain-coordinates-in-python
    '''
    x, y = xy
    inner = a * (x - x0)**2
    inner += 2 * b * (x - x0)**2 * (y - y0)**2
    inner += c * (y - y0)**2
    return amp * np.exp(-inner)



def test_2d_gaussfit():
    # from https://stackoverflow.com/questions/27539933/2d-gaussian-fit-for-intensities-at-certain-coordinates-in-python
    np.random.seed(1977) # For consistency
    x, y = np.random.random((2, 10))
    xy = np.vstack([x,y])
    x0, y0 = 0.3, 0.7
    amp, a, b, c = 1, 2, 3, 4
    true_params = [amp, x0, y0, a, b, c]    
    zobs = gauss2d(xy, amp, x0, y0, a, b, c)

    i = zobs.argmax()
    guess = [1, x[i], y[i], 1, 1, 1]
    pred_params, uncert_cov = curve_fit(gauss2d, xy, zobs, p0=guess)

    zpred = gauss2d(xy, *pred_params)
    print('True parameters: ', true_params)
    print('Predicted params:', pred_params)
    print('Residual, RMS(obs - pred):', np.sqrt(np.mean((zobs - zpred)**2)))

    # from here its plotting:

    fig, ax = plt.subplots()
    scat = ax.scatter(x, y, c=zobs, s=200)
    fig.colorbar(scat)
    plt.show()

    yi, xi = np.mgrid[:1:30j, -.2:1.2:30j]
    xyi = np.vstack([xi.ravel(), yi.ravel()])

    zpred = gauss2d(xyi, *pred_params)
    zpred.shape = xi.shape

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=zobs, s=200, vmin=zpred.min(), vmax=zpred.max())
    im = ax.imshow(zpred, extent=[xi.min(), xi.max(), yi.max(), yi.min()],
                   aspect='auto')
    fig.colorbar(im)
    ax.invert_yaxis()
    plt.show()

def main():

    # generate some data
    # change the parameters as you see fit
    y = two_gauss_func([20,20,3,60,50,4],(np.arange(50)/.5))
    x = (np.arange(50)/.5)
    plt.plot(x,y)
    data = np.asarray([x,y])
#    do_two_gauss_fit(data,verbose = True)
    
    do_multi_gauss_fit(data,nopeaks = 2,verbose = True)
    

    

if __name__ == "__main__":
    main()
