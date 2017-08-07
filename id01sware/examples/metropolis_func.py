#!/usr/bin/env python
from id01lib.ptycho import plotting
from id01lib.ptycho import metropolis_func as mf

xx=[0,50,5]
yy=[0,50,10]
spiral_step = 5
meshx,meshy = mf.spiral_mesh_MIR(xx,yy,spiral_step,square_aperture=True)
#meshx,meshy=standard_mesh(xx,yy)
# number of cities the salesman must visit

x,y,ordering,iteration_history, energy_history,kT_history = mf.simulated_anneal(meshx,meshy)
plotting.plot_result(x,y,ordering,iteration_history, energy_history,kT_history)

