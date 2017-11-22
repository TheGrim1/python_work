from __future__ import print_function
import sys, os
import h5py
import numpy as np
import time

#### Uses id01 sware courtesy of S. Leake
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript/id01sware/"))
import id01lib.ptycho.scan_utils as id01su



def do_randomized_spiral_scan(self, *p, **kw):
    print("credits to S.Leake and id01sware")
    try:
        wds = p[0].split()
        if len(wds) not in (12,13):
            print("usage: randomized_spiral_scan <xrange> <yrange>")
            return
        
    except:
        ptb()
   


def randomized_spiral(diameter, spiral_step, square_aperture=False, randomization_factor = 0.2):
    '''
    Uses id01 sware courtesy of S. Leake
    diameter = diameter
    spiral_step = distance between each point
    the number of points scales with diameter * diameter (duh)
    start_th = change the location of your first point
    square_aperture = spiral scan that fills a square aperture
    randomization_factor controls randomization amplitude: factor * spiral_step = amplitude of random field
    """
    '''
    
    
    x = y = [xmin, xmax, xstep]
    
    spiral = np.asarray(id01su.spiral_mesh_MIR(x,y,spiral_step,square_aperture))
    random_field =  randomization_factor * (np.random.random_sample(spiral.shape)-0.5)
    coords = spiral + random_field
    xmesh = coords[0]
    ymesh = coords[1]
    
    return xmesh, ymesh
        
def main():

    import matplotlib.pyplot as plt

    print('green virgin spiral')
    print('red: randomize with 0.05 of a step')
    print('blue: randomize with 0.2 of a step')
    g = np.asarray(id01su.spiral_mesh_MIR((0,10,10),(0,10,10),0.3))
    r = np.asarray(id01su.spiral_mesh_MIR((0,10,5),(0,10,10),0.3)) + 0.05 * (np.random.random_sample(g.shape)-0.5)
    b = randomized_spiral((0,10,10),(0,10,10),0.3, randomization_factor = 0.2)
    plt.plot(g[0],g[1],'go')
    plt.plot(r[0],r[1],'ro')
    plt.plot(b[0],b[1],'bo')
    plt.show()

    print('This is MAIN') 


    


if __name__=='__main__':
    main()    
