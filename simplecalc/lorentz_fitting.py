# based on gauss_fitting

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
from simplecalc.fitting import *


def conservative_peak_guess(data,
                            nopeaks = 2,
                            verbose = False,
                            plot = False,
                            threshold = 10,
                            minwidth = 2,
                            maxwidth = None):
    '''
    trying to find good starting positions for the lorentz fitting:
    print("p[0], a1: ", v[0])
    print("p[1], mu1: ", v[1])
    print("p[2], sigma1: ", v[2])
    print("p[3], a2: ", v[3])
    print("p[4], mu2: ", v[4])
    print("p[5], sigma2: ", v[5])
    '''
    if maxwidth == None:
        width  = len(data[0])/40.0
    else:
        width = maxwidth
    widths = []
    peaks = []
    while len(peaks) < nopeaks and width > minwidth:
        widths.append(width)
        peaks = find_peaks(data[1],widths = np.asarray(widths))
        width *= 0.5
    # fix error when no peaks are found:
    if len(peaks)<nopeaks:
        if verbose:
            print('only found {} peaks, extending list with evenly placed peaks to {}'.format(len(peaks),nopeaks))
        missing_nopeaks = nopeaks - len(peaks)
        default_peaks = np.zeros(nopeaks,dtype = np.int32)
        for i,peak in enumerate(peaks):
            default_peaks[i]=peak
        for i,j in enumerate(range(len(peaks),nopeaks)):
            default_peaks[j] = int((0.5+i)*len(data[0])/missing_nopeaks)
        peaks = default_peaks

        
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


def multi_lorentz_func(p,t,nopeaks):
    function = np.zeros(shape = (len(t)))
    for i in range(nopeaks):
        function += lorentz_func(p[list(range(i*3,(i+1)*3))],t)
    return function


def multi_lorentz_residual(p,x,y,nopeaks):
    return multi_lorentz_func(p,x,nopeaks = nopeaks) - y


def do_variable_lorentz_fit(data, v0= None, plot = False, verbose = False, minwidth = 2, maxwidth = None):
    '''
    Fits with nopeaks = len(v0)/ number of lorentz functions, if v==None, fits 3 peaks.
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
        return multi_lorentz_residual(p = p, x = x, y = y, nopeaks=nopeaks)
    out = leastsq(optfunction, v0[:], args=(data[0], data[1]), maxfev=100000, full_output=1) #Lorentz Fit
    v = out[0] #fit parameters out
    covar = out[1] #covariance matrix output
    xxx = np.arange(min(data[0]),max(data[0]),data[0][1]-data[0][0])
    ccc = multi_lorentz_func(v,xxx,nopeaks = nopeaks) # this will only work if the units are pixel and not wavelength

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


def do_variable_lorentzbkg_pipeline(data,
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
        peaks, residual = do_variable_lorentz_fit(data=data,
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

def do_iterative_variable_lorentzbkg_pipeline(data,
                                              nopeaks =3,
                                              plot = False,
                                              verbose = False,
                                              threshold = 10,
                                              minwidth = 20,
                                              maxwidth = None):

    new_data=np.copy(data)
    found_peaks_no = 0
    peaks_left = np.copy(nopeaks)
    found_peaks=[]
    
    while peaks_left>0:

        peaks, residual = do_variable_lorentzbkg_pipeline(data=new_data,
                                                          nopeaks = nopeaks,
                                                          plot = plot,
                                                          verbose = verbose,
                                                          threshold = threshold,
                                                          minwidth = minwidth,
                                                          maxwidth = maxwidth)

        peaks_found_no = sum(np.where(np.asarray(peaks)!=0.0,1,0)[:,0])

        for i in range(peaks_found_no):
            new_data[1] -= lorentz_func(peaks[i],data[0])
            found_peaks.append(list(peaks[i]))
            
        peaks_left -= max(peaks_found_no,1)

        

    
    found_peaks = [x for x in found_peaks if (x[1]>min(data[0]) and x[1]<max(data[0]))]
    for i in range(nopeaks):
        found_peaks.append([0,0,0])
    found_peaks.sort()
    found_peaks=found_peaks[::-1]
    found_peaks = np.asarray(found_peaks)
    
    found_peaks = found_peaks[0:nopeaks]
    found_peaks[:,2]=np.abs(found_peaks[:,2])
    
    if plot:
        plt.plot(data[0],data[1],'black')

    if verbose == True:
        print('found peaks at:')
        
        for i, peak in enumerate(found_peaks):
            print('peak {}: a {}, mu {}, sig {}'.format(i, peak[0], peak[1], peak[2]))
            if plot == True:
                plt.plot(data[0],lorentz_func(peak,data[0]))
                if peak[1] < max(data[0]) and peak[1]> min(data[0]):
                    plt.axvline(x = peak[1],color = 'red')
    if plot == True:
        plt.plot(data[0],new_data[1],'--')
        plt.show()

        
    return found_peaks, residual


def multi_lorentz_and_poly_func(p,t, no_peaks):
    '''
    p[0:3],p[3:6],p[6:9],p[9:12],p[12:15]the lorentzes
    p0 = a
    p1 = mu
    p2 = sigma (0.5 * fwhm)
    etc.
    p[15:] = the polynomial_func
    '''
    p=np.asarray(p)
    function = multi_lorentz_func(p[0:no_peaks*3],t,no_peaks) + polynomial_func(p[no_peaks*3::],t)
    return function

def multi_lorentz_and_poly_residual(p,x,y,no_peaks):
    return multi_lorentz_and_poly_func(p,x,no_peaks = no_peaks) - y

def do_multi_lorentz_and_poly_fit(data, no_peaks, poly_degree, verbose = False, lorentz_index_guess=None, prefit=False):

    datashape = data.shape

    poly_guess = [0]*(poly_degree+1)
    if poly_degree >0:
        slope_guess, offset_guess = do_lower_linear_guess(data,20)
        # # DEBUG
        # plt.plot(data[:,0],data[:,1])
        # plt.plot(data[:,0],linear_func([slope_guess, offset_guess], data[:,0]))
        # plt.show()
        poly_guess[-2] = slope_guess
    else:
        offset_guess = data[:,1].min()
        slope_guess= 0
        
    poly_guess[-1] = offset_guess

    
    lorentz_guess=[]
    if lorentz_index_guess == None:
        # equidistant peak spacing
        lorentz_index_guess = [x*int(datashape[0]/no_peaks) + int(datashape[0]/no_peaks*2) for x in range(no_peaks)]
               
    for i in range(no_peaks):
        index = lorentz_index_guess[i]
        lorentz_guess.append(data[index,1]-linear_func([slope_guess, offset_guess], data[index,0]))
        lorentz_guess.append(data[index,0])
        lorentz_guess.append(3*np.absolute(data[0,0]-data[1,0]))

    par_guess = lorentz_guess + poly_guess


    
    # # DEBUG:
    # for i in range(len(par_guess)):
    #     print(par_guess[i])
    # plt.plot(data[:,0], linear_func(np.asarray([slope_guess, offset_guess]), data[:,0]),'-r')
    # plt.plot(data[:,0], data[:,1], data[:,0], multi_lorentz_and_poly_func(np.asarray(par_guess), data[:,0], no_peaks))
    # plt.show()
    
    if prefit:
        if verbose:
            print('performing prefit with fixed peak positions')
        par_guess = do_multi_fixed_lorentz_and_poly_fit(data,
                                                        fixed_lorentz_index=None,
                                                        poly_degree = poly_degree,
                                                        verbose = verbose,
                                                        par_guess=par_guess)
    
    def optfunction(p,x,y):
        return multi_lorentz_and_poly_residual(p = p, x = x, y = y, no_peaks=no_peaks)

    if verbose:
        print('performing actual fit with free peak positions')
    out = leastsq(optfunction, par_guess, args=(data[:,0], data[:,1]), maxfev=100000, full_output=1)
    beta = out[0]
    # sigma, max non negative:
    for i, par in enumerate(beta[0:3*no_peaks]):
        if i%3!=1:
            beta[i] = np.abs(beta[i])
    
    if verbose:
        fig, ax = plt.subplots()
        
        for i,res in enumerate(beta):
            print('beta[{}] = {}'.format(i,beta[i]))
        ax.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0], multi_lorentz_and_poly_func(beta, data[:,0], no_peaks), "r--", lw = 2)
        plt.tight_layout()

        plt.show()
        
    return beta

def multi_fixed_lorentz_and_poly_func(p,t, fixed_pos):
    '''
    p[0:3],p[3:6],p[6:9],p[9:12],p[12:15]the lorentzes
    p0 = a
    p1 = mu
    p2 = sigma (0.5 * fwhm)
    etc.
    p[15:] = the polynomial_func
    '''
    no_peaks = len(fixed_pos)
    beta = [0]*(len(p)+no_peaks)
    # sigma non negative and collect fixed parameters into beta:
    beta[0:3*no_peaks:3] = p[0:2*no_peaks:2]
    beta[1:3*no_peaks+1:3] = fixed_pos
    beta[2:3*no_peaks+2:3] = p[1:2*no_peaks+1:2]
    beta[3*no_peaks:] = p[2*no_peaks:]

    beta=np.asarray(beta)
    
    function = multi_lorentz_func(beta[0:no_peaks*3],t,no_peaks) + polynomial_func(beta[no_peaks*3::],t)
    return function

def multi_fixed_lorentz_and_poly_residual(p,x,y,fixed_pos):
    return multi_fixed_lorentz_and_poly_func(p,x,fixed_pos=fixed_pos) - y

def do_multi_fixed_lorentz_and_poly_fit(data,
                                        fixed_lorentz_index=[1,2,3],#or par_guess
                                        poly_degree = 2,
                                        verbose = False,
                                        par_guess = None): #or fixed_lorentz_index
    '''
    receives either a full par_guess:

    or just the fixed pos, and then does own par_guess
    '''

    datashape = data.shape
                                        
    if par_guess==None:
        no_peaks = len(fixed_lorentz_index)
        poly_guess = [0]*(poly_degree+1)
        if poly_degree >0:
            slope_guess, offset_guess = do_lower_linear_guess(data,20)
            # # DEBUG:
            # plt.plot(data[:,0],data[:,1])
            # plt.plot(data[:,0],linear_func([slope_guess, offset_guess], data[:,0]))
            # plt.show()
            poly_guess[-2] = slope_guess
        else:
            offset_guess = data[:,1].min()
            slope_guess= 0
        
        poly_guess[-1] = offset_guess
        lorentz_guess=[]
                       
        for i in range(no_peaks):
            index = fixed_lorentz_index[i]
            lorentz_guess.append(data[index,1]-linear_func([slope_guess, offset_guess], data[index,0]))
            lorentz_guess.append(data[index,0])
            lorentz_guess.append(3*np.absolute(data[0,0]-data[1,0]))
            
        par_guess = lorentz_guess + poly_guess
    else: #par_guess is used
        no_peaks = int(len(par_guess)-poly_degree-1)/3
    

    # seperate fixed from fit parameters
    fixed_pos = []
    fit_par_guess = []
    for i,par in enumerate(par_guess[:no_peaks*3]):
        if i%3==1:
            fixed_pos.append(par)
        else:
            fit_par_guess.append(par)

    [fit_par_guess.append(x) for x in par_guess[-poly_degree-1:]]
            
    def optfunction(p,x,y):
        return multi_fixed_lorentz_and_poly_residual(p = p, x = x, y = y, fixed_pos=fixed_pos)
    
    out = leastsq(optfunction, fit_par_guess, args=(data[:,0], data[:,1]), maxfev=100000, full_output=1)
    
    fit_par_result = out[0]
    beta = [0]*(3*no_peaks + poly_degree+1)
    # sigma non negative and collect fixed parameters into beta:
    beta[0:3*no_peaks:3] = np.abs(fit_par_result[0:2*no_peaks:2])
    beta[1:3*no_peaks:3] = fixed_pos
    beta[2:3*no_peaks:3] = np.abs(fit_par_result[1:2*no_peaks+1:2])
    beta[-poly_degree-1:] = fit_par_result[-poly_degree-1:]
    
    if verbose :
        fig, ax = plt.subplots()
        
        for i,res in enumerate(beta):
            print('beta[{}] = {}'.format(i,beta[i]))
        ax.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0], multi_lorentz_and_poly_func(beta, data[:,0], no_peaks), "r--", lw = 2)
        plt.tight_layout()

        plt.show()
        
    return beta

def five_lorentz_and_poly(p,t):
    '''
    p[0:3],p[3:6],p[6:9],p[9:12],p[12:15]the lorentzes
    p0 = a
    p1 = mu
    p2 = sigma (0.5 * fwhm)
    etc.
    p[15:] = the polynomial_func
    '''
    return lorentz_func(p[0:3],t) + lorentz_func(p[3:6],t) + lorentz_func(p[6:9],t) + lorentz_func(p[9:12],t) + lorentz_func(p[12:15],t) + polynomial_func(p[15:],t)

def do_five_lorentz_and_poly_fit(data, poly_degree, verbose = False, lorentz_index_guess=None):
    '''
    old: use mulit_lorentz
    '''
    Model = scipy.odr.Model(five_lorentz_and_poly)
    Data = scipy.odr.RealData(data[:,0], data[:,1])

    datashape = data.shape
    index_spread = [x*int(datashape[0]/5) + int(datashape[0]/10) for x in range(5)]
    lorentz_guess = []

    # slope_guess = (data[-1,1]-data[0,1])/(data[-1,0]-data[0,0])
    # offset_guess = (-data[0,0]*slope_guess) + data[0,1]
    slope_guess = 0
    offset_guess=min(data[:,1])

    poly_guess = [0]*(poly_degree+1)
    poly_guess[-1] = offset_guess
    # poly_guess[-2] = slope_guess
        
    if lorentz_index_guess == None:
        for i in range(5):
            lorentz_guess.append(data[index_spread[i],1]-linear_func([slope_guess, offset_guess], data[i,0]))
            lorentz_guess.append(15*np.absolute(data[0,0]-data[1,0]))
            lorentz_guess.append(data[index_spread[i],0])
    else:
        for i in range(5):
            index = lorentz_index_guess[i]
            lorentz_guess.append(data[index,1]-linear_func([slope_guess, offset_guess], data[i,0]))
            lorentz_guess.append(15*np.absolute(data[0,0]-data[1,0]))
            lorentz_guess.append(data[index,0])

    Odr = scipy.odr.ODR(Data, Model, lorentz_guess + poly_guess, maxit = 10000000)
    Odr.set_job(fit_type=2)
    output = Odr.run()
    #output.pprint()
    beta    = output.beta
    betastd = output.sd_beta

    # sigma non negative:
    beta = np.asarray([beta[0],beta[1],abs(beta[2]),beta[3],beta[4],abs(beta[5]),beta[6],beta[7],abs(beta[8]),beta[9],beta[10],abs(beta[11]),beta[12],beta[13],abs(beta[14])] +list(beta[15:]))
    
    if verbose :
        fig, ax = plt.subplots()
        
        for i,res in enumerate(beta):
            print('beta[{}] = {}'.format(i,beta[i]))
        ax.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0],five_lorentz_and_poly(beta, data[:,0]), "r--", lw = 2)

        plt.tight_layout()

        plt.show()
        
    return beta

def four_lorentz_and_poly(p,t):
    '''
    old: use mulit_lorentz
    p[0:3],p[3:6],p[6:9],p[9:12] the lorentzes
    p0 = a
    p1 = mu
    p2 = sigma (0.5 * fwhm)
    etc.
    p[12:] = the polynomial_func
    '''
    return lorentz_func(p[0:3],t) + lorentz_func(p[3:6],t) + lorentz_func(p[6:9],t) + lorentz_func(p[9:12],t) + polynomial_func(p[12:],t)

def do_four_lorentz_and_poly_fit(data, poly_degree, verbose = False, lorentz_index_guess=None):
    
    Model = scipy.odr.Model(four_lorentz_and_poly)
    Data = scipy.odr.RealData(data[:,0], data[:,1])

    datashape = data.shape
    index_spread = [x*int(datashape[0]/4) + int(datashape[0]/8) for x in range(4)]
    lorentz_guess = []

    # slope_guess = (data[-1,1]-data[0,1])/(data[-1,0]-data[0,0])
    # offset_guess = (-data[0,0]*slope_guess) + data[0,1]
    slope_guess = 0
    offset_guess=min(data[:,1])

    poly_guess = [0]*(poly_degree+1)
    poly_guess[-1] = offset_guess
    # poly_guess[-2] = slope_guess
        
    if lorentz_index_guess == None:
        for i in range(4):
            lorentz_guess.append(data[index_spread[i],1]-linear_func([slope_guess, offset_guess], data[i,0]))
            lorentz_guess.append(30*np.absolute(data[0,0]-data[1,0]))
            lorentz_guess.append(data[index_spread[i],0])
    else:
        for i in range(4):
            index = lorentz_index_guess[i]
            lorentz_guess.append(data[index,1]-linear_func([slope_guess, offset_guess], data[i,0]))
            lorentz_guess.append(30*np.absolute(data[0,0]-data[1,0]))
            lorentz_guess.append(data[index,0])

    Odr = scipy.odr.ODR(Data, Model, lorentz_guess + poly_guess, maxit = 10000000)
    Odr.set_job(fit_type=2)
    output = Odr.run()
    #output.pprint()
    beta    = output.beta
    betastd = output.sd_beta

    # sigma non negative:
    beta = np.asarray([beta[0],beta[1],abs(beta[2]),beta[3],beta[4],abs(beta[5]),beta[6],beta[7],abs(beta[8]),beta[9],beta[10],abs(beta[11])] +list(beta[12:]))
    
    if verbose :
        fig, ax = plt.subplots()
        
        for i,res in enumerate(beta):
            print('beta[{}] = {}'.format(i,beta[i]))
        ax.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0],four_lorentz_and_poly(beta, data[:,0]), "r--", lw = 2)

        plt.tight_layout()

        plt.show()
        
    return beta


def lorentz_and_poly_func(p,t):
    '''
    p0 = a
    p1 = mu
    p2 = sigma (0.5 * fwhm)
    p3 = offset
    p4 = slope
    etc.
    '''
    return p[0]*p[2]**2 /((t-p[1])**2 + p[2]**2) + polynomial_func(p[3:],t)

def do_lorentz_and_poly_fit(data, poly_degree, verbose = False):
    
    Model = scipy.odr.Model(lorentz_and_poly_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    a_guess = np.max(data[:,1])
    sigma_guess = 50*np.absolute(data[0,0]-data[1,0])
    mu_guess    = data[:,0][np.argmax(data[:,1])]
    
    Odr = scipy.odr.ODR(Data, Model, [a_guess,mu_guess, sigma_guess]+(poly_degree+1)*[0], maxit = 10000000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta    = output.beta
    betastd = output.sd_beta

    if verbose :
        fig, ax = plt.subplots()
    #    print "poly", fit_np
        print("fit result a: \n", beta[0])
        print("fit result mu: \n", beta[1])
        print("fit result sigma: \n", beta[2])
        print("fit result slope: \n", beta[3])
        print("fit result offset: \n", beta[4])

        ax.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0], lorentz_and_poly_func(beta, data[:,0]), "r--", lw = 2)

        plt.tight_layout()

        plt.show()
        
    return beta

def lorentz_func(p, t):
    '''
    p0 = maximum # positive!
    p1 = mu
    p2 = sigma (0.5 * fwhm)
    '''
    return abs(p[0])*(p[2]**2) /((t-p[1])**2 + p[2]**2)

def do_lorentz_fit(data, verbose = False):
    
    Model = scipy.odr.Model(lorentz_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    a_guess = np.max(data[:,1])
    sigma_guess = 50*np.absolute(data[0,0]-data[1,0])
    mu_guess    = data[:,0][np.argmax(data[:,1])]

    Odr = scipy.odr.ODR(Data, Model, [a_guess,mu_guess, sigma_guess], maxit = 10000000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta    = output.beta
    betastd = output.sd_beta

    if verbose :
        fig, ax = plt.subplots()
    #    print "poly", fit_np
        print("fit result a: \n", beta[0])
        print("fit result mu: \n", beta[1])
        print("fit result sigma: \n", beta[2])

        ax.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0], lorentz_func(beta, data[:,0]), "r--", lw = 2)

        plt.tight_layout()

        plt.show()
        
    return [beta[0],beta[1],abs(beta[2])]
