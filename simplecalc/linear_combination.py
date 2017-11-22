from __future__ import print_function

import sys,os
import numpy as np
import scipy.odr
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.figsize': [12.0,10.0]})
import math
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import simplecalc.gauss_fitting as gauss_fit
import simplecalc.calc as calc


def component_func(p, t, components, force_positive = True):

    if force_positive:
        p = np.abs(p)
    
    y = np.zeros(shape = t.shape)
    for i in range(len(components[0,:])-1):
        y += p[i]*np.interp(t,components[:,0],components[:,i+1])
        
    return y



def do_component_analysis(data, components, verbose = False, force_positive = True, normalize = True):
    '''
    fits data with a linear combination of components
    data.shape = (l, 2) (2d) 
    components.shape = (r,n+1), components [:,0] = xaxis
    returns unnormalize vector of each components content in data, len vector = n
    '''

    if verbose:
        plt.plot(data[:,0],data[:,1],color = 'blue', linewidth = 4)
        for i in range(1,len(components[0,:])):
            plt.plot(components[:,0],components[:,i],color = 'red', linewidth = 1)
        plt.show()
            
    def fit_func(p,t):
        return component_func(p, t, components, force_positive)

    
    Model = scipy.odr.Model(fit_func)
    Data = scipy.odr.RealData(data[:,0], data[:,1])
    startguess = [0.5]*(len(components[0,:])-1)

    Odr = scipy.odr.ODR(Data, Model, startguess , maxit = 1000000)
    Odr.set_job(fit_type=2)    
    output = Odr.run()
    #output.pprint()
    beta     = output.beta
    betastd  = output.sd_beta
    residual = fit_func(beta, data[:,0]) - data[:,1]

    if force_positive:
        beta = np.abs(beta)
    
    if verbose:
        ax1 = plt.gca()
        print("force_positive = %s" % force_positive)
        print("Relative weight of composition found: ")
        print(beta)
        ax1.plot(data[:,0], data[:,1], linewidth = 4,color = 'black')                    
        color = ['r','g','b','darkblue','grey']
        for i in range(1,len(components[0,:])):
#            ax1.plot(components[:,0], components[:,i] * beta[i-1],color = color[i-1],linewidth = 2)
            ax1.plot(components[:,0], components[:,i] * beta[i-1], linewidth = 2)
        ax1.plot(data[:,0], residual, color = 'grey',linewidth = 2)
        ax1.set_ylabel('normalized signal [norm.]')
        ax1.set_xlabel('energy [eV]')
        ax1.set_xticklabels(['{:d}'.format(int(x)) for x in ax1.get_xticks()])
#        ax1.legend(['data','GaAs','Ga-metal','alpha-Ga2O3','beta-Ga2O3','residual'])
        ax1.set_ylim([-0.5,2])
#        plt.tight_layout()
        plt.show()
    if normalize:
        beta = beta / np.sum(beta)   
    return beta, residual

def test():

    ### setup test data
    data = np.zeros(shape = (50,2))
    data[:,0] = np.linspace(0, 2 * np.pi, 50)
    data[:,1] = np.sin(data[:,0])+1

    xaxis = np.atleast_1d(np.linspace(0, 2 * np.pi, 50))

    p = []
    components = np.zeros(shape = (len(xaxis),7))
    components[:,0] = np.linspace(0, 2 * np.pi, 50)

    for i in range(6):
        p.append([1,i,2])   
        components[:,i+1] = gauss_fit.gauss_func(p[i],xaxis)

    plt.plot(data[:,0],data[:,1],linewidth = 4)                    
    plt.plot(xaxis,components[:,1:7])
    plt.show()

    ###run test
    
    vector = do_component_analysis(data, components, verbose = True)

    
if __name__ == "__main__":
    test()
