''' Demonstration of the Metropolis algorithm (simulated annealing)
    for solution of the "traveling salesman" problem.

    R.G. Erdmann, 2007
    PSF compatible license

    Given some number of (randomly placed in this example) cities, one
    of which is designated as the "home" city, the problem is to find
    the optimum ordering of cities starting at the home city and
    ending at the home city, where optimum is defined as "lowest
    lowest possible overall distance traveled".  The solution is
    achieved by use of the Metropolis algorithm.

    Note: The implementation provided here is not particularly
    efficient and is only intended to give a simple demonstration of
    how the Metropolis method (a.k.a. simulated annealing) works for
    solving a combinatorial optimization problem.  In particular, the
    recomputation of the entire path length when swapping two cities
    is extremely inefficient compared to the ideal implementation,
    especially for problems in which the number of cities is large.
    Furthermore, the details of the temperature schedule and initial
    scaling could be modified to improve the convergence rate, but
    optimum or near-optimum solutions are still reached quite quickly
    with the current demonstration implementation.

    Dependencies: 
    ptych_scan_utils.py

    Example:
    python metropolis_func.py
'''

import numpy as np
#from scipy import *
from .scan_utils import *

def path_length(x,y,ordering):
    '''Return the total path length of some ordering of cities.

    The cities are traversed in a loop, so the total length is
    independent of the starting city.  Hence, one need not know the
    starting ("home") city to compute the overall path length.

    (Path length will serve as the energy in the Metropolis method.)
    '''
    pos = np.c_[x[ordering], y[ordering]]
    length = np.sum(np.sqrt(np.sum((pos - np.roll(pos, -1, axis=0))**2, axis=1)))
    return length
    
    
def perturb_ordering(i, j, ordering):
    '''Reversibly perturb the ordering.

    "Reversibly" as defined here means that repetition of the
    perturb_ordering operator with the same i,j leaves the system in
    its original state.

    '''
    # the following perturbation reverses the order of cities between i and j
    lo = min(i, j)
    hi = max(i, j)
    ordering[lo:hi] = ordering[lo:hi].copy()[::-1]

    # comment out the above 3 lines and uncomment the following to
    # simply swap pairs of cities as the perturbation.
    #ordering[i], ordering[j] = ordering[j], ordering[i]

    
def simulated_anneal(x,y,kTmult=0.99995,kT = 10):
    NUM_CITIES = np.prod(x.shape)
    print(x.shape)
    print(NUM_CITIES)
    # randomly pick x, y coordinates for the cities from a uniform [0,1)
    # distribution, i.e., pick random locations within a unit square.
    x = x.flatten()
    y = y.flatten()

    #HOME_CITY = 0
    
    # As an initial ordering, we just go to the cities in order.
    ordering = np.arange(NUM_CITIES) 

    E_current = path_length(x,y,ordering)

    # scaled dimensionless initial temperature.
    # Note that the initial thermal energy scale kT should be on the same
    # order as the overall configuration energy scale deltaE, where deltaE
    # is the energy difference between the initial and optimum energies.
    iteration = 0
    i = 0
    j = 0

    # keep lists of the system energy and kT as we cool
    energy_history = []
    kT_history = []
    iteration_history = []

    #============================================================
    # Main cooling loop
    #============================================================

    while True:
        # Pick two cities at random.  Note that since we are assuming a
        # round-trip starting from the home city and ending at the home
        # city, we don't need to keep the home city in the first slot
        # since ultimately we're just going to draw a closed loop for the
        # round trip.
        i = np.random.random_integers(0, NUM_CITIES-1)
        j = np.random.random_integers(0, NUM_CITIES-1)
        if i == j:
            # try again
            continue

        # now (possibly temporarily) perturb the city ordering
        perturb_ordering(i, j, ordering)
        
        # the new configuration has an energy
        E_proposed = path_length(x,y,ordering)
        
        if E_proposed <= E_current:
            # we always accept the new ordering if it has a lower energy
            E_current = E_proposed
        else:
            # the proposed perturbation would increase the energy
            delta_E = E_proposed - E_current
            # Probability of acceptance is P_accept, computed according to
            # the Boltzmann distribution
            P_accept = np.exp(-delta_E / kT)
            r = np.random.uniform() # draw a uniformly distributed random
                                 # number from 0 to 1
            if r < P_accept:
                # accept the new ordering
                E_current = E_proposed
            else:
                # reject the new ordering, so put the system back in its
                # pre-perturbation configuration
                perturb_ordering(i, j,ordering)

        kT *= kTmult#0.9995 # cool *very* slowly. (0.9995 was determined to work well with 60 cities.)

        iteration += 1

        if iteration % 10 == 0: # print status every 10 steps
            print('%6d: kT: %1.3e  E:%1.3e' % (iteration, kT, E_current))
            energy_history.append(E_current)
            kT_history.append(kT)
            iteration_history.append(iteration)
        
        if kT < 1e-3: # stop cooling once the system is very cold.
            break

    print("Starting distance: ", E_proposed, "Final distance: ",  E_current)
    return x,y,ordering,iteration_history, energy_history,kT_history


