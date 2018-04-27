#!/usr/bin/env python
import StringIO
from id01lib.ptycho.scan_utils import *
from id01lib.ptycho import plotting
## range of initial point generation 
x=[0,125,1]# min,max,step
# about 26 points along the defined axis if you use [0,250] across range scale accordingly to your needs
y=[0,125,1]

## define the position in real life that you want the scan to take place
centre=[23.16,100.3]
stroke=[.7,.7] # if you have an asymmetric beam these shouldn't be equal
xlims=[centre[0]-stroke[0]/2-1,centre[0]+stroke[0]/2+1]
ylims=[centre[1]-stroke[1]/2-1,centre[1]+stroke[1]/2+1]
window=True
shortest_distance = False#True

## generate meshes (select what you want to use)
## spirals
spiral_step = 3
#meshx,meshy=spiral_mesh(x,y,spiral_step,square_aperture=window)
#meshx,meshy=spiral_mesh_MIR(x,y,spiral_step,square_aperture=window)
meshx,meshy,maxdist,max_xx,max_yy=fermat_spiral_mesh(x,y,spiral_step,square_aperture=window,return_max_step = True)
#meshx,meshy=fermat_spiral_mesh(x,y,spiral_step)
#meshx,meshy=fermat_spiral_golden_mesh(x,y,spiral_step,square_aperture=window)
## concentric
rad_step=5
points_inner_circle=5
#meshx,meshy=concentric_mesh(x,y,rad_step,points_inner_circle,square_aperture=window)
## simple
#meshx,meshy=triangular_mesh(x)
#meshx,meshy=random_mesh(x,y)
#meshx,meshy=diamond_mesh(x)
#meshx,meshy=standard_mesh(x,y)

print("Largest step...")
print("dim 1: %.2dnm"%(scale_mesh_dist(meshx,max_xx,stroke[0],centre[0])/1e-3))
print("dim 2: %.2dnm"%(scale_mesh_dist(meshy,max_yy,stroke[1],centre[1])/1e-3))
## apply scaling
meshx=scale_mesh(meshx,stroke[0],centre[0])
meshy=scale_mesh(meshy,stroke[1],centre[1])
meshx,meshy=window_ptych(meshx,meshy,xlims,ylims)
print("number of points for spec scan: %i "%meshx.shape)#,meshy.shape)


## lets have a look
plotting.plot_mesh(meshx,meshy,savefile='') # put savefile to save image

## calc shortest motor movement distances
if shortest_distance:
	meshx,meshy=sim_anneal_distance(meshx,meshy)

## save the .dat file for spec
fh = StringIO.StringIO()
save_ptych_scan(meshx,meshy,fh)
fh.seek(0)
print(fh.read())



