'''
see also gauss_fitting.py
here: 
do_linear_fit(data)
do_quadratic_fit(data)
do_cubic_fit(data)
do_exp_fit(data)
do_gauss_fit(data)
do_logistic_fit(data)
'''
from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.odr
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import math


def do_lower_linear_guess(data, no_points):
    '''
    provides a lower estimate for a linear background usefull as starting parameters for fitting complicated functions
    evenly distributes data[:,0] into no_points and chooses min slope between these points
    and a corresponding offset so that data - linear fit is allways positive
    '''
    
    datalength = len(data[:,0])
    selected_indexes = [x*int(datalength/no_points) for x in range(no_points)]
    selected_indexes.append(datalength-1)
    points = data[selected_indexes,:]
    # sort according to value:

    switch = zip(points[:,1],points[:,0])
    switch.sort()
    switch = np.asarray(switch)

    points = np.asarray(zip(switch[:,1],switch[:,0]))

    slopes = [abs((points[0,1]-points[x+1,1])/(points[0,0]-points[x+1,0])) for x in range(no_points-1)]

    slope = min(slopes)
    slope = np.sign(data[-1,1] - data[0,1])*slope

    
    offset_guess = (-points[0,0]*slope) + points[0,1]
    
    residual = data[:,1] - linear_func([slope, offset_guess], data[:,0])
    offset = offset_guess + residual.min()

    # #DEBUG:
    # plt.plot(data[:,0],residual,'-y')
    # plt.plot(data[:,0],data[:,1],data[:,0],linear_func([slope, offset], data[:,0]))
    # plt.plot(points[:,0],points[:,1],'rx')
    # plt.show()
    
    return slope, offset
        

def empirical_func(p, t, data):
    y = p[0]*np.interp(t+p[2],data[:,0],data[:,1]) + p[1]
    return y

def do_empirical_fit(data, empirical_function, verbose=False):
    '''
    tested
    fit function is A0 * (empirical_function[t + A2]) + A1 with 
    returns (A0,A1,A2) = beta
    '''
    def fit_function(p,t):
        return empirical_func(p,t,empirical_function)

    _func = fit_function
    Model = scipy.odr.Model(_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    startguess = [1,0,0] # no changes

    Odr = scipy.odr.ODR(Data, Model, startguess , maxit = 1000000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta     = output.beta
    betastd  = output.sd_beta
    residual = _func(beta, data[:,0]) - data[:,1]

    if verbose:
        fig, ax = plt.subplots()
    #    print "poly", fit_np
        print("fit result amp: \n", beta[0])
        print("fit result offset: \n", beta[1])
        print("fit result shift: \n", beta[2])

        ax.plot(data[:,0], data[:,1], "bo")
        # plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0], _func(beta, data[:,0]), "r--", lw = 2)
        # ax.plot(data[:,0], _func([max_guess, min_guess, inflection_guess, sigma_guess ], data[:,0]), "g--", lw = 2)

        plt.tight_layout()

        plt.show()
        
    return beta

def guess_period_scan(data, period_guess=None, verbose=False):

    test_data = data[:,1]-data[:,1].mean()
    if type(period_guess)==type(None):
        scan_range = np.linspace(1,data.shape[0]/2,data.shape[0]*2)
    else:
        scan_range = np.asarray(period_guess)
    correlation=[]
    for period in scan_range:
        p_correlation=0
        for phase in np.linspace(0,360,72):
            probe = sin_func([1,phase,period,0],data[:,0])
            p_correlation=max(np.sum(probe*test_data),p_correlation)
        correlation.append(p_correlation)
    if verbose:
        plt.plot(scan_range,correlation)
    return scan_range[np.asarray(correlation).argmax()] 


def sin_func(p, t):
    '''
    p[0] = amp
    p[1] = phase in deg
    p[2] = period
    p[3] = offset
    p[0]*np.sin(t - p[1]) + p[2]
    '''
    return np.abs(p[0])*np.sin((t/np.abs(p[2])+p[1]/360)*2*np.pi) + p[3]

def do_sin_fit(data, verbose = False,period_guess=None):

    _func = sin_func
    Model = scipy.odr.Model(_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    amp_guess = np.max(data[:,1]) - np.min(data[:,1])
    phase_guess = 0.0
    offset_guess = 0.5*(np.max(data[:,1]) + np.min(data[:,1]))
    if type(period_guess) == type('asdf'):
        if period_guess.upper() == 'SCAN':
            period_guess = guess_period_scan(data,period_guess=None,verbose=verbose)
        else:
            period_guess = float(period_guess)
    if type(period_guess)==list:
        period_guess = guess_period_scan(data,period_guess=period_guess,verbose=verbose)
    if type(period_guess)==np.ndarray:
        period_guess = guess_period_scan(data,period_guess=period_guess,verbose=verbose)
    if type(period_guess)==type(None):
        period_guess=0
    
    if verbose:
        print('amp_guess = ', amp_guess)
        print('phase_guess = ', phase_guess)
        print('period_guess = ', period_guess)
        print('offset_guess = ', offset_guess)

    guess = [amp_guess, phase_guess, period_guess, offset_guess]
    Odr = scipy.odr.ODR(Data, Model, guess, maxit = 10000000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta    = output.beta
    betastd = output.sd_beta

    beta[0] = np.abs(beta[0])
    beta[2] = np.abs(beta[2])

    if verbose :
        fig, ax = plt.subplots()
    #    print "poly", fit_np
        print("fit result amp: \n", beta[0])
        print("fit result phase: \n", beta[1])
        print('fit_result period = ', beta[2])
        print("fit result offset: \n", beta[3])

        ax.plot(data[:,0], _func(beta, data[:,0]), "r--", lw = 2)

        ax.plot(data[:,0], data[:,1], "bo")
        # plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)

        # ax.plot(data[:,0], _func([max_guess, min_guess, inflection_guess, sigma_guess ], data[:,0]), "g--", lw = 2)

        plt.tight_layout()

        plt.show()
        
    return beta

def sin360period_func(p, t):
    '''
    p[0] = amp
    p[1] = phase in deg
    p[2] = offset
    p[0]*np.sin(t - p[1]) + p[2]
    '''
    return np.abs(p[0])*np.sin((t+p[1])/360*2*np.pi) + p[2]

def do_sin360period_fit(data, verbose = False):

    _func = sin360period_func
    Model = scipy.odr.Model(_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    amp_guess = np.max(data[:,1]) - np.min(data[:,1])
    phase_guess = 0.0
    offset_guess = 0.5*(np.max(data[:,1]) + np.min(data[:,1]))

    
    if verbose:
        print('amp_guess = ', amp_guess)
        print('phase_guess = ', phase_guess)
        print('offset_guess = ', offset_guess)

    guess = [amp_guess, phase_guess, offset_guess]
    Odr = scipy.odr.ODR(Data, Model, guess, maxit = 10000000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta    = output.beta
    betastd = output.sd_beta

    beta[0] = np.abs(beta[0])

    if verbose :
        fig, ax = plt.subplots()
    #    print "poly", fit_np
        print("fit result amp: \n", beta[0])
        print("fit result phase: \n", beta[1])
        print("fit result offset: \n", beta[2])

        ax.plot(data[:,0], _func(beta, data[:,0]), "r--", lw = 2)

        ax.plot(data[:,0], data[:,1], "bo")
        # plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)

        # ax.plot(data[:,0], _func([max_guess, min_guess, inflection_guess, sigma_guess ], data[:,0]), "g--", lw = 2)

        plt.tight_layout()

        plt.show()
        
    return beta



def general_logistic_func(p, t):
    '''
    only works for "increasing" function
    p[0] = max
    p[1] = min
    p[2] = inflection point
    p[3] = sigma (approx)
    '''
    
    return p[0]/ (1 + math.e**(-(0.5*np.pi/p[3])*(t-p[2]))) + p[1]

def do_logistic_fit(data, verbose = False):
    '''
    only works for "increasing" function
    p[0] = max
    p[1] = min
    p[2] = inflection point
    p[3] = sigma (approx)
    '''
    
    _func = general_logistic_func
    Model = scipy.odr.Model(_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    max_guess = np.max(data[:,1])
    min_guess = np.min(data[:,1])
    inflection_guess = np.mean(data[:,0])
    sigma_guess = 1
    if verbose:
        print('max_guess = ', np.max(data[:,1]))
        print('min_guess = ', np.min(data[:,1]))
        print('inflection_guess = ', np.mean(data[:,0]))
        print('steepness_guess = ', 1)
    Odr = scipy.odr.ODR(Data, Model, [max_guess, min_guess, inflection_guess, sigma_guess ], maxit = 10000000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta    = output.beta
    betastd = output.sd_beta

    if verbose :
        fig, ax = plt.subplots()
    #    print "poly", fit_np
        print("fit result max: \n", beta[0])
        print("fit result min: \n", beta[1])
        print("fit result inflection point: \n", beta[2])
        print("fit result sigma: \n", beta[3])

        ax.plot(data[:,0], data[:,1], "bo")
        # plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0], _func(beta, data[:,0]), "r--", lw = 2)
        # ax.plot(data[:,0], _func([max_guess, min_guess, inflection_guess, sigma_guess ], data[:,0]), "g--", lw = 2)

        plt.tight_layout()

        plt.show()
        
    return beta



def error_func(p,t):
    # save the sign of t
    sign = np.where(t >= 0, 1, -1)
    t = abs(t)

    # constants for approximation of error function
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    a6  =  0.3275911

    
    # A&S formula 7.1.26
    r = 1.0/(1.0 + a6*t)
    y = 1.0 - (((((a5*r + a4)*r) + a3)*r + a2)*r + a1)*r*math.exp(-t*t)

    return sign*y # erf(-x) = -erf(x)


def gauss_func(p, t):
    '''
    p0 = a
    p1 = mu
    p2 = sigma
    '''
    return p[0]*1/(math.sqrt(2*math.pi*(p[2]**2)))*math.e**((-(t-p[1])**2/(2*p[2]**2)))

def do_gauss_fit(data, verbose = False,sigma_in_pxl=50):
    
    Model = scipy.odr.Model(gauss_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    a_guess = np.max(data[:,1])
    sigma_guess = sigma_in_pxl*np.absolute(data[0,0]-data[1,0])
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
        ax.plot(data[:,0], gauss_func(beta, data[:,0]), "r--", lw = 2)

        plt.tight_layout()

        plt.show()
        # raw_input('next')
        
    return beta

def linear_func(p, t):
    return p[0] * t + p[1]

def do_linear_fit(data, verbose = False):
    
    Model = scipy.odr.Model(linear_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    Odr = scipy.odr.ODR(Data, Model, [-2, 1], maxit = 10000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta    = output.beta
    betastd = output.sd_beta

    
    if verbose :
        fig, ax = plt.subplots()
    #    print "poly", fit_np
        print("fit result [pxl/frame]: \n", beta[0])

        ax.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0], linear_func(beta, data[:,0]), "r--", lw = 2)

        plt.tight_layout()

        plt.show()
    return beta
    
def quadratic_func(p, t):
    return p[0]*t**2 + p[1]*t + p[2]

def do_quadratic_fit(data, verbose = False):
    
    Model = scipy.odr.Model(quadratic_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    Odr = scipy.odr.ODR(Data, Model, [-10, -10, -2], maxit = 1000000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta = output.beta
    betastd = output.sd_beta
#    print "poly", fit_np


    if verbose:
        fig, ax = plt.subplots()
        print("fit result y = %s x2 + %s x + %s  " % (beta[0],beta[1],beta[2]))
        ax.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0], quadratic_func(beta, data[:,0]), "r--", lw = 2)
        plt.tight_layout()
        plt.show()

    return beta


def cubic_func(p, t):
    return p[0]*t**3 + p[1]*t**2 + p[2]*t + p[3]

def do_cubic_fit(data, verbose=False):
    
    Model = scipy.odr.Model(cubic_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    Odr = scipy.odr.ODR(Data, Model, [-10, -10, -2, 1], maxit = 1000000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta = output.beta
    betastd = output.sd_beta

    if verbose:
        fig, ax = plt.subplots()
    #    print "poly", fit_np
        print("fit result y = %s x3 + %s x2 + %s x + %s  " % (beta[0],beta[1],beta[2], beta[3]))
        ax.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0], cubic_func(beta, data[:,0]), "r--", lw = 2)
        plt.tight_layout()
        plt.show()

    return beta

def polynomial_func(beta, t):
    '''
    test : works
    '''
    p = beta[::-1]
    return np.polynomial.polynomial.polyval(t,p)

def do_polynomial_fit(data, degree, verbose = False):
    '''
    test works
    '''
    print('data in fitting:')
    print('x:')
    print(data[:,0])
    print('y:')
    print(data[:,1]) 
    
    p = np.polynomial.polynomial.polyfit(data[:,0], data[:,1], degree)
    beta = p[::-1]

    print('result of fitting:')
    print('x:')
    print(data[:,0])
    print('y:')
    print(polynomial_func(beta, data[:,0]))

    if verbose:
        fig, ax = plt.subplots()
        print('fount polynomial coefficients an ... a0:')
        print(beta)
        ax.plot(data[:,0], data[:,1], "bo")
        ax.plot(data[:,0], polynomial_func(beta, data[:,0]), "r--", lw = 2)
        plt.tight_layout()
        plt.show()
    return beta 
        
def exp_func(p, t):
    return p[0] * math.e**(t*p[1]) + p[2]

def do_exp_fit(data, verbose = False):
    
    Model = scipy.odr.Model(exp_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    Odr = scipy.odr.ODR(Data, Model, [+10, -0.0010, 3], maxit = 1000000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta = output.beta
    betastd = output.sd_beta
#    print "poly", fit_np

    if verbose:
        fig, ax = plt.subplots()
        print("fit result y = %s e^(x * %s) + %s" % (beta[0],beta[1],beta[2]))
        ax.plot(data[:,0], data[:,1], "bo")
        #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        ax.plot(data[:,0], exp_func(beta, data[:,0]), "r--", lw = 2)
        plt.tight_layout()
        plt.show()

    return beta



def main(row = [1353., 1353., 1353.],
         col = [980., 987., 995.],
         frame = [87,86,85],
         ):
    
    dist = [math.sqrt(row[i]**2 + col[i]**2) for i in range(len(row))]
    data = np.zeros(shape=(len(row),2))
    data[:,0] = np.array(frame) 
    data[:,1] = np.array(dist)
    
    do_linear_fit(data,verbose=True)


if __name__ == "__main__":
    main()
