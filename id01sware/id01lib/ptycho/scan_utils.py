###################
# Library of functions to generate ptycho scans
# S.J.Leake 2014/12/15
###################
# TODO:
#
# spiral DONE
# fermat spiral DONE
# concentric circles DONE
# triangular DONE
# standard mesh DONE
# diamond mesh DONE
# noisy everything DONE
# optimisation for reduced movements - DONE - I used simulated annealing to optimise the distances
# random walk - started

# 20150810 added maxdist in x/y for overlap consideration only to fermat_spiral_mesh

import numpy as np

###########
# ptycho mesh scans
###########

# archimedes spiral
# NB to say such a spiral has a constant distance between successive turns is false
# according to parallel curves perpendicular to the spiral itself it is not the case
# radially from the origin it is constant 
def spiral_mesh(x,y,spiral_step,start_th = np.pi,square_aperture=False):
    """
    x/y = x/y[min,max,step]
    spiral_step = distance between each point
    start_th = change the location of your first point
    square_aperture = spiral scan that fills a square aperture
    """
    # centre
    cen = [x[0]+(x[1]-x[0])/2,y[0]+(y[1]-y[0])/2]
    # If you begin at theta zero moving in an anticlockwise direction
    # the pixel furthest away is bottom right of your FOV i.e. pix_br
    pix_tl = [x[0],y[0]]
    pix_br = [x[1],y[1]]
    max_r = np.sqrt((pix_br[0]-cen[0])**2+(pix_br[1]-cen[1])**2)
    spiral_th = start_th
    spiral_r = 0
    meshx = np.array([]);meshy = np.array([])
    while spiral_r< max_r:
        xx=spiral_r*np.cos(spiral_th)+cen[0]
        yy=spiral_r*np.sin(spiral_th)+cen[1]
        #print xx,yy
        # Calculate the next coordinates
        spiral_r =(spiral_step*spiral_th)/(2*np.pi)
        # If these coordinates lie within the defined rectangle
        # then store them
        if square_aperture:
            # logic for a defined aperture 
            if pix_tl[0]<=xx<=pix_br[0] and pix_tl[1]<=yy<=pix_br[1]:
                meshx=np.r_[meshx,xx]
                meshy=np.r_[meshy,yy]
        else:
            # standard spiral
            meshx=np.r_[meshx,xx]
            meshy=np.r_[meshy,yy]
        # Update the step in angle based on current angle etc.
        spiral_th += (2*np.pi)/np.sqrt(1+spiral_th**2)
    return meshx,meshy

# From MIR depends on your starting position - the overlap of the central parts. 
def spiral_mesh_MIR(x,y,spiral_step,square_aperture=False):
    """
    xy = xy[min,max,step]
    """
    # centre
    cen = [x[0]+(x[1]-x[0])/2,y[0]+(y[1]-y[0])/2]
    # If you begin at theta zero moving in an anticlockwise direction
    # the pixel furthest away is bottom right of your FOV i.e. pix_br
    pix_tl = [x[0],y[0]]
    pix_br = [x[1],y[1]]
    max_r = np.sqrt((pix_br[0]-cen[0])**2+(pix_br[1]-cen[1])**2)
    spiral_th = np.pi
    spiral_r = 0
    meshx = np.array([])
    meshy = np.array([])
    while spiral_r< max_r:
        xx=spiral_r*np.cos(spiral_th)+cen[0]
        yy=spiral_r*np.sin(spiral_th)+cen[1]
        #print xx,yy
        # Calculate the next coordinates
        spiral_r =(spiral_step*spiral_th)/(2*np.pi)
        # If these coordinates lie within the defined rectangle
        # then store them
        if square_aperture:
            # logic for a defined aperture 
            if pix_tl[0]<=xx<=pix_br[0] and pix_tl[1]<=yy<=pix_br[1]:
                meshx=np.r_[meshx,xx]
                meshy=np.r_[meshy,yy]
        else:
            # standard spiral
            meshx=np.r_[meshx,xx]
            meshy=np.r_[meshy,yy]
        # Update the step in angle based on current angle etc.
        spiral_th += (2*np.pi)/np.sqrt(1+spiral_th**2)
    return meshx,meshy

# fermat spiral (a la X. Huang)
def fermat_spiral_mesh(x,y,spiral_step,square_aperture=False, return_max_step = False):
    """
    xy = xy[min,max,step]
    """
    # centre
    cen = [x[0]+(x[1]-x[0])/2,y[0]+(y[1]-y[0])/2]
    # If you begin at theta zero moving in an anticlockwise direction
    # the pixel furthest away is bottom right of your FOV i.e. pix_br
    pix_tl = [x[0],y[0]]
    pix_br = [x[1],y[1]]
    max_r = np.sqrt((pix_br[0]-cen[0])**2+(pix_br[1]-cen[1])**2)
    phi_zero = np.pi# 0#(1+np.sqrt(5))/2.
    spiral_th = phi_zero
    spiral_r = 0
    meshx = np.array([]); meshy = np.array([])
    i=0

    xx0,yy0 = cen[0],cen[1]
    maxdist=0
    max_xx=0
    max_yy=0
    while spiral_r < max_r:
        # Calculate the next coordinates
        spiral_r = np.sqrt(spiral_step**2*spiral_th)#np.sqrt((spiral_step**2/phi_zero)*spiral_th)/(2*np.pi) # this is effectively sqrt(s_step**2*i)
        spiral_r = (spiral_step**2*spiral_th)/(2*np.pi)
        # Update the step in angle based on current angle etc.
        xx = spiral_r*np.cos(spiral_th)+cen[0]
        yy = spiral_r*np.sin(spiral_th)+cen[1]
        # find distance to previous point and log max
        dist = ((xx-xx0)**2+(yy-yy0)**2)**0.5
        #print dist, xx, yy
        deltaxx=xx-xx0
        deltayy=yy-yy0
        if dist > maxdist and pix_tl[0]<=xx<=pix_br[0] and pix_tl[1]<=yy<=pix_br[1]:
            maxdist = dist

        if deltaxx > max_xx and pix_tl[0]<=xx<=pix_br[0]:
            max_xx = deltaxx

        if deltayy > max_yy and pix_tl[1]<=yy<=pix_br[1]:
            max_yy = deltayy
        # update last point reference
        xx0=xx
        yy0=yy
	
        #print spiral_th,spiral_r
        # If these coordinates lie within the defined rectangle
        # then store them
        if square_aperture:
            # logic for a defined aperture 
            if pix_tl[0]<=xx<=pix_br[0] and pix_tl[1]<=yy<=pix_br[1]:
                meshx=np.r_[meshx,xx]
                meshy=np.r_[meshy,yy]
        else:
            # standard spiral
            meshx=np.r_[meshx,xx]
            meshy=np.r_[meshy,yy]
        i+=1
        spiral_th += (2*np.pi)/np.sqrt(1+spiral_th**2) #phi_zero
    
    #print maxdist
    #print max_xx, max_yy
    if return_max_step:
        return meshx,meshy,maxdist,max_xx,max_yy
    else:
        return meshx,meshy
    
# fermat spiral r = c*n^1/2
def fermat_spiral_golden_mesh(x,y,spiral_step,square_aperture=False):
    """
    xy = xy[min,max,step]
    """
    # centre
    cen = [x[0]+(x[1]-x[0])/2,y[0]+(y[1]-y[0])/2]
    # If you begin at theta zero moving in an anticlockwise direction
    # the pixel furthest away is bottom right of your FOV i.e. pix_br
    pix_tl = [x[0],y[0]]
    pix_br = [x[1],y[1]]
    max_r = np.sqrt((pix_br[0]-cen[0])**2+(pix_br[1]-cen[1])**2)
    spiral_th = 0
    spiral_r = 0
    meshx = np.array([])
    meshy = np.array([])
    i=0
    while spiral_r < max_r:
        xx=spiral_r*np.cos(spiral_th)+cen[0]
        yy=spiral_r*np.sin(spiral_th)+cen[1]
        #print xx,yy
        # Calculate the next coordinates
        spiral_r =(spiral_step*np.sqrt(i))
        # If these coordinates lie within the defined rectangle
        # then store them
        if square_aperture:
            # logic for a defined aperture 
            if pix_tl[0]<=xx<=pix_br[0] and pix_tl[1]<=yy<=pix_br[1]:
                meshx=np.r_[meshx,xx]
                meshy=np.r_[meshy,yy]
        else:
            # standard spiral
            meshx=np.r_[meshx,xx]
            meshy=np.r_[meshy,yy]
        i+=1
        # Update the step in angle based on current angle etc.
        spiral_th += (1+np.sqrt(5))/2.
    return meshx,meshy
    
    # concentric circle - for a rather

def concentric_mesh(x,y,rad_step,points_inner_circle,square_aperture=False):
    """
    x/y = x/y[min,max,step]
    rad_step = radial step
    points_inner_circle = points in inner circle, choose depending on symmetry you want to induce 5+ is generally better
    """
    cen = [x[0]+(x[1]-x[0])/2,y[0]+(y[1]-y[0])/2]
    pix_tl = [x[0],y[0]]
    pix_br = [x[1],y[1]]
    max_r = np.sqrt((pix_br[0]-cen[0])**2+(pix_br[1]-cen[1])**2)
    nr = 1+ np.floor(max_r/rad_step)
    meshx = np.array([])
    meshy = np.array([])
    for i in np.arange(0,nr+1):
        rr=i*rad_step
        delta_th = 2*np.pi/(i*points_inner_circle)
        #print delta_th
        for j in np.arange(i*points_inner_circle):
            th = j*delta_th
            xx =cen[0]+rr*np.sin(th)
            yy =cen[1]+rr*np.cos(th) 
            if square_aperture:
                # logic for a defined aperture 
                if pix_tl[0]<=xx<=pix_br[0] and pix_tl[1]<=yy<=pix_br[1]:
                    meshx = np.r_[meshx,xx]
                    meshy = np.r_[meshy,yy]
            else:
                # standard spiral
                meshx = np.r_[meshx,xx]
                meshy = np.r_[meshy,yy]
    return meshx,meshy
    
# triangular mesh 
def triangular_mesh(xy):
    """
    xy = xy[min,max,step]
    """
    atoms = [[0,0],[0.5,0.5/np.tan(np.degrees(30))]]
    meshx = np.array([])
    meshy = np.array([])
    min=xy[0];max=xy[1]
    step=xy[2]*2
    for i in np.arange(min,max,step):
        for j in np.arange(min,max,step/(np.tan(np.degrees(30)))): 
            for k in atoms:
                meshx=np.r_[meshx,i+k[0]*step]
                meshy=np.r_[meshy,j+k[1]*(step)]
    return meshx,meshy

# random mesh 
def random_mesh(x,y):
    """
    a few caveats with this one 
    # the overlap parameter goes out the window
    # you need a fill in function at the end to catch missing data
    """
    step = [(x[1]-x[0])/x[2],  (y[1]-y[0])/y[2]]
    a = np.array([np.random.rand(np.prod(step))])
    b = np.array([np.random.rand(np.prod(step))])
    
    # calculate coverage - depends on footprint
    # generate a 2D array and fill with ones where you get a hit
    
    # fill in the gaps
    
    return a,b
    
# diamond mesh
def diamond_mesh(xy):
    """
    xy = xy[min,max,step]
    """
    atoms = [[0,0],[0.5,0.5]]
    meshx = np.array([])
    meshy = np.array([])
    min=xy[0];max=xy[1]
    step=xy[2]*2
    for i in np.arange(min,max,step):
        for j in np.arange(min,max,step): 
            for k in atoms:
                meshx=np.r_[meshx,i+k[0]*step]
                meshy=np.r_[meshy,j+k[1]*step]
    return meshx,meshy

# standard mesh
def standard_mesh(x,y):
    """
    x = x[min,max,step]
    y = y[min,max,step]
    """
    meshx,meshy = np.meshgrid(np.arange(x[0],x[1],x[2]),np.arange(y[0],y[1],y[2]))
    return meshx,meshy

# export ptych_scan in .dat format to be read in using spec
def save_ptych_scan(x,y,out_file):
    """
    x = array of x positions
    y = array of y positions 
    
    output - x,y ASCII file of positions 
    """
    np.savetxt(out_file,np.c_[x.flatten(),y.flatten()],fmt = '%.5f')

    
 ###########
 # other funcs
 ###########
 
def minimise_movements(meshx,meshy):
    """
    invoke the metropolis algorithm to reduce total distance moved and accentuate errors
    N. Metropolis J Chem Phys 21, 1087-1092 (1953)
    Numerical recipes in C 
    X. Huang Optics Express 22 10 2014
    i.e. what is the minimum hamiltonian path
    """
    # if you have a desired motor direction you can optimise accordingly
    # need a monte carlo optimisation
    #b=np.c_[meshx,meshy,np.sqrt(meshx**2+meshy**2)]
    #b=np.c_[meshx,meshy]#,np.sqrt(meshx**2+meshy**2)]
    #c=list(tuple(b))
    #np.savetxt(b,'test.dat') # hack to get a structured array fast
    #b=np.loadtxt('test.dat',dtype={'names': ('meshx', 'meshy', 'r'),'formats':('f8','f8','f8')})
    
    ##########
    #    this is too slow need a speed up
    """
    import itertools as it
    import math

    def dist(x,y):
        return math.hypot(y[0]-x[0],y[1]-x[1])

    paths = [ p for p in it.permutations(c)]
    path_distances = [ sum(map(lambda x: dist(x[0],x[1]),zip(p[:-1],p[1:]))) for p in paths ]
    min_index = argmin(path_distances)

    print paths[min_index], path_distances[min_index]
    """
    ##########

    return meshx,meshy
    
def sim_anneal_distance(meshx,meshy):
    from . import metropolis_func as mf
    x, y, ordering, iteration_history, energy_history, kT_history = mf.simulated_anneal(meshx,meshy)
    return x[ordering],y[ordering]
    
def poisson_motors(meshx,meshy,error):
    """
    meshx,meshy - SE
    error = a reasonable amount of positioning error e.g np.sqrt(step)
    """
    meshx+=(np.random.random(meshx.shape)-0.5)*error
    meshy+=(np.random.random(meshy.shape)-0.5)*error
    return meshx,meshy
    
def calc_overlap(beam,alpha,x,y):
    """
    beam = [h,v] (microns)
    alpha = incident angle (degrees)
    xypos = [meshx,meshy] (microns)
    """
    # overlap can be described in two ways
    # as a f(diameter), f(area)
    # in general we don't have a uniform spot due to alpha therefore an overlap 
    # should be described in both dimensions
    footprint = [beam[0],beam[1]/np.abs(np.sin(np.degrees(alpha)))]
    step_sep = [x[2],y[2]]
    # diameter
    print("overlap x:", (footprint[0]-step_sep[0])/footprint[0]*100, "overlap y:",(footprint[1]-step_sep[1])/footprint[1]*100)
    # for general interest the area overlap can be calculated as follows
    print("area_olp x:", (circle_circle_intersection(footprint[0]/2.,footprint[0]/2.,step_sep[0])/(np.pi*((footprint[0]/2.)**2)))*100, \
    "area_olp y:", (circle_circle_intersection(footprint[1]/2.,footprint[1]/2.,step_sep[1])/(np.pi*((footprint[1]/2.)**2)))*100)

def calc_overlap_2D_arr_pts(beam,alpha,meshx,meshy):
    """
    CALCULATE THE OVERLAP OF AN ENTIRE SCAN - NOT SIMPLE
    break it down into pixel overlaps, need a minimum overlap of 60% 
    therefore multiply each dimension by 3, fill in with ones and search for zeroes or ones
    Does not help when you have non integer positions
    beam = [h,v] (microns)
    alpha = incident angle (degrees)
    xypos = [meshx,meshy] (microns)
    """
    # overlap can be described in two ways
    # as a f(diameter), f(area)
    # in general we don't have a uniform spot due to alpha therefore an overlap 
    # should be described in both dimensions
    footprint = [beam[0],beam[1]/np.abs(np.sin(np.degrees(alpha)))]
    step_sep = [x[2],y[2]]
    # diameter
    print("overlap x:", (footprint[0]-step_sep[0])/footprint[0]*100, "overlap y:",(footprint[1]-step_sep[1])/footprint[1]*100)
    # for general interest the area overlap can be calculated as follows
    print("area_olp x:", (circle_circle_intersection(footprint[0]/2.,footprint[0]/2.,step_sep[0])/(np.pi*((footprint[0]/2.)**2)))*100, \
    "area_olp y:", (circle_circle_intersection(footprint[1]/2.,footprint[1]/2.,step_sep[1])/(np.pi*((footprint[1]/2.)**2)))*100)
    
def circle_circle_intersection(r1,r2,d):
    area = (r1**2)*np.arccos((d**2+r1**2-r2**2)/(2*d*r1))\
            + (r2**2)*np.arccos((d**2+r2**2-r1**2)/(2*d*r2))\
            - 0.5*np.sqrt((r1+r2-d)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
    return area


def scale_mesh(mesh,stroke,centre):
    return mesh/(mesh.max()-mesh.min())*stroke+centre-stroke/2

def scale_mesh_dist(mesh,dist,stroke,centre):
    #print  mesh.max(),mesh.min()
    return dist/(mesh.max()-mesh.min())*stroke

def window_ptych(meshx,meshy,xlims,ylims):
    wtype = np.dtype([('x',meshx[0].dtype),('y',meshy[0].dtype)])
    arr_meshpoints=np.empty(meshx.shape,dtype=wtype)
    arr_meshpoints['x']=meshx
    arr_meshpoints['y']=meshy
    #xlims = [125,375]
    #ylims = [100,400]
    a=arr_meshpoints[((arr_meshpoints['x']>xlims[0]) & (arr_meshpoints['x']<xlims[1]) & (arr_meshpoints['y']>ylims[0]) & (arr_meshpoints['y']<ylims[1]))]
    return a['x'],a['y']

