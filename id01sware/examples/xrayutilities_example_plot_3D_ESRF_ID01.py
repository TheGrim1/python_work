#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of xrayutilities.
#
# xrayutilities is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2012-2013 Dominik Kriegner <dominik.kriegner@gmail.com>

# ALSO LOOK AT THE FILE xrayutilities_id01_functions.py
# Edited by SJL 20150828

# built-ins
from __future__ import print_function
import os
import sys
#sys.path.append(os.path.join(os.path.expanduser("~"), "id01sware"))

# community
import numpy as np
import xrayutilities as xu
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from id01lib import xrayutilities_id01_functions as id01
    from id01lib import hdf5_writer as h5w
except ImportError:
    print("trying local import.")
    sys.path.append(os.path.join(os.path.abspath(os.pardir)))
    from id01lib import xrayutilities_id01_functions as id01
    from id01lib import hdf5_writer as h5w


home = "DATADIR"  # data path (root)
datadir = '/data/visitor/hc2615/id01/detector/E16034/' # data path for CCD/Maxipix files
specdir = '/data/visitor/hc2615/id01/spec'  # location of spec file

# to put in hdf5 file
scan_nos=[5,10,12,15,34,43,52,61,70, 79, 88, 110, 119, 128, 137, 146, 155]
data_fn = 'E16034'

h5w.create_hdf5(specdir+'/E16034.spec',image_prefix='data_mpx4_',imagedir=datadir,scan_nos=scan_nos,output_fn=data_fn,sigfig="%05d")


# scan to plot
SCANNR = 155


h5file = data_fn + '.h5'#os.path.join(specdir, data_fn + ".h5")



# number of points to be used during the gridding
nx, ny, nz = 100,100,100

qx, qy, qz, gint, gridder = id01.gridmap(
	    h5file, SCANNR, nx, ny, nz,angdelta=[0, 0, 0, 0])

# ################################################
# for a 3D plot using python function i sugggest
# to use mayavi's mlab package. the basic usage
# is shown below. otherwise have a look at the
# file xrayutilities_export_data2vtk.py in order learn
# how you can get your data to a vtk file for further
# processing.
# #####
# one of the following import statements is needed
# depending on the system/distribution you use
from mayavi import mlab	
# from enthough.mayavi import mlab
# plot 3D map using mayavi mlab
QX,QY,QZ = np.mgrid[qx.min():qx.max():1j * nx,
                      qy.min():qy.max():1j * ny,
                      qz.min():qz.max():1j*nz]
INT = xu.maplog(gint,4.5,0)
mlab.figure()
mlab.contour3d(QX, QY, QZ, INT, contours=15, opacity=0.5)
mlab.colorbar(title="log(int)", orientation="vertical")
mlab.axes(nb_labels=5, xlabel='Qx', ylabel='Qy', zlabel='Qz')
mlab.title('SCAN:%i'%SCANNR, size=0.4)

mlab.show()
# mlab.close(all=True)
############################################







"""
print gint.shape
# plot 2D sums using matplotlib
plt.figure()
plt.contourf(qz, qy, np.log(gint.sum(axis=0)))
plt.xlabel(r"QZ ($1/\AA$)")
plt.ylabel(r"QY ($1/\AA$)")
plt.colorbar()
plt.savefig(os.path.join("pics","filename.png"))

# plot 2D slice using matplotlib
plt.figure()
plt.contourf(qy, qx, np.log(gint.sum(axis=2)))
plt.xlabel(r"QX ($1/\AA$)")
plt.ylabel(r"QY ($1/\AA$)")
plt.colorbar()
plt.savefig(os.path.join("pics","filename1.png"))

# plot 2D slice using matplotlib
plt.figure()
plt.contourf(qz, qx, np.log(gint.sum(axis=1)))
plt.xlabel(r"QZ ($1/\AA$)")
plt.ylabel(r"QX ($1/\AA$)")
plt.colorbar()
plt.savefig(os.path.join("pics","filename2.png"))

"""

# coordinates:          QX,QY,QZ
# integrated intensity: gint

#integrating 
"""
data=np.where((QZ>4.56)*(QZ<4.598),gint,0)
data=np.where((QX>0.05)*(QX<0.15)*(QY>0.06)*(QY<0.15),0,data)

data2D = np.sum(data,axis=2)

import pylab as pl
pl.pcolormesh(QX[:,:,0],QY[:,:,0],data2D,)
pl.show()

data2D.sum().sum()
"""
"""
def crop_domain(gint,minmax,scan):
  data=np.where((QZ>minmax[4])*(QZ<minmax[5]),gint,0)
  data=np.where((QX>minmax[0])*(QX<minmax[1])*(QY>minmax[2])*(QY<minmax[3]),0,data)
  data2D= data.sum(axis=2)
  import pylab as pl
  pl.pcolormesh(QX[:,:,0],QY[:,:,0],data2D,)
  pl.savefig("domain_pop_0045_%i"%(scan))
  pl.clf()
  #pl.show()
  print scan, " : ",data2D.sum().sum()



scans = [55,]
minmaxvals = {55:[-0.05,0.05,-0.05,0.05,4.69,4.74]}
h5file = data_fn + '.h5'#os.path.join(specdir, data_fn + ".h5")



scan=55
qx, qy, qz, gint, gridder = id01.gridmap(
	    h5file, scan, nx, ny, nz,angdelta=[-0.02, 1.9, 0 ,0])#0.262,-0.626]) # eta/phi/nu/del offsets
QX,QY,QZ = np.mgrid[qx.min():qx.max():1j * nx,
                      qy.min():qy.max():1j * ny,
                      qz.min():qz.max():1j*nz]

#crop_domain(gint,minmaxvals[scan],scan)
minmax=minmaxvals[scan]
data=np.where((QZ>minmax[4])*(QZ<minmax[5]),gint,0)
data=np.where((QX>minmax[0])*(QX<minmax[1])*(QY>minmax[2])*(QY<minmax[3]),0,data)
data2D= data.sum(axis=2)
import pylab as pl
pl.pcolormesh(QX[:,:,0],QY[:,:,0],data2D,)
#pl.savefig("domain_pop_0045_%i"%(scan))
#pl.clf()
pl.show()
print data2D.sum().sum()
"""

"""
# number of points to be used during the gridding
nx, ny, nz = 100,100,100

#0.03 -0.06
for scan in scans:
   qx, qy, qz, gint, gridder = id01.gridmap(
	    h5file, scan, nx, ny, nz,angdelta=[0, 1.9, 0 ,0]) # eta/phi/nu/del offsets
   QX,QY,QZ = np.mgrid[qx.min():qx.max():1j * nx,
                      qy.min():qy.max():1j * ny,
                      qz.min():qz.max():1j*nz]

   crop_domain(gint,minmaxvals[scan],scan)

"""
  

