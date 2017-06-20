'''
see also gauss_fitting.py
here: 
do_linear_fit(data)
do_quadratic_fit(data)
do_cubic_fit(data)
do_exp_fit(data)
'''

import numpy as np
import scipy.odr
import matplotlib.pyplot as plt
import math

def linear_func(p, t):
    return p[0] * t + p[1]

def do_linear_fit(data, verbose = False):
    
    Model = scipy.odr.Model(linear_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    Odr = scipy.odr.ODR(Data, Model, [-2, 1], maxit = 10000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta = output.beta
    betastd = output.sd_beta

    if verbose :    
    #    print "poly", fit_np
        print "fit result [pxl/frame]: \n", beta[0]

        plt.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        plt.plot(data[:,0], linear_func(beta, data[:,0]), "r--", lw = 2)

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
    if vebose:
        print "fit result y = %s x2 + %s x + %s  " % (beta[0],beta[1],beta[2])
        plt.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        plt.plot(data[:,0], quadratic_func(beta, data[:,0]), "r--", lw = 2)
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
    #    print "poly", fit_np
        print "fit result y = %s x3 + %s x2 + %s x + %s  " % (beta[0],beta[1],beta[2], beta[3])
        plt.plot(data[:,0], data[:,1], "bo")
    #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        plt.plot(data[:,0], cubic_func(beta, data[:,0]), "r--", lw = 2)
        plt.tight_layout()
        plt.show()

    return beta

def polynomial_func(beta, t):
    '''
    test : failed  TODO
    '''
    
    p = beta[::-1]
    return np.polynomial.polynomial.polyval(p,t)

def do_polynomial_fit(data, degree, verbose = False):
    '''
    test : failed  TODO
    '''
    print('data in fitting:')
    print'x:'
    print data[:,0]
    print'y:'
    print data[:,1] 
    
    p = np.polynomial.polynomial.polyfit(data[:,0], data[:,1], degree)
    beta = p[::-1]

    print('result of fitting:')
    print'x:'
    print data[:,0]
    print'y:'
    print polynomial_func(beta, data[:,0])

    if verbose:
        print('fount polynomial coefficients an ... a0:')
        print(beta)
        plt.plot(data[:,0], data[:,1], "bo")
        plt.plot(data[:,0], polynomial_func(beta, data[:,0]), "r--", lw = 2)
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

        print "fit result y = %s e^(x * %s) + %s" % (beta[0],beta[1],beta[2])
        plt.plot(data[:,0], data[:,1], "bo")
        #    plt.plot(data[:,0], numpy.polyval(fit_np, data[:,0]), "r--", lw = 2)
        plt.plot(data[:,0], exp_func(beta, data[:,0]), "r--", lw = 2)
        plt.tight_layout()
        plt.show()

    return beta

def main(row = [1365., 1365., 1365.],
         col = [1632., 1638.5, 1644.5],
         frame = [37,36,35],
         ):
    
    dist = [math.sqrt(row[i]**2 + col[i]**2) for i in range(len(row))]
    data = np.zeros(shape=(len(row),2))
    data[:,0] = np.array(frame) 
    data[:,1] = np.array(dist)
    
    do_linear_fit(data)


if __name__ == "__main__":
    main()
