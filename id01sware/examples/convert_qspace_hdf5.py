# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import xrayutilities as xu
from id01lib.plot import Normalize, DraggableColorbar
from id01lib.xrd import qconversion, geometries, detectors

h5f = h5py.File("/data/id01/inhouse/crichter/tmp/edo/hc2912.h5", "r")

sample = "E16095_furnace"



# compulsory parameters:
scanno = 9 # This will be translated to the entry `9.1` in the hdf5 file
cen_pix_hor  = 281.847 # center channel of the Maxipix detector
cen_pix_vert = 278.792 # center channel of the Maxipix detector
det_distance = 0.422 # from logbook, in meters

geometry = qconversion.ID01psic() # that's the default

# optional parameters
# kmap = None # not used yet. an integer number here in case the kmaps should be processed
nav = None #[1,1] # reduce data: number of pixels to average in each detector direction
roi = None #[0,516,0,516] # define a region of interest, 0-516 is the whole thing
bins = [-2,-2,-2] # in reciprocal space. Multiples of minimum. [-1,-1,-1] for highest resolution
monitor = "exp1" # try =None if normalization seems wrong
#monitor = None

#geometry.set_offsets(rhx=0, rhy=0, phi=0) # example to correct sample tilt if needed (rhx~-eta at phi=0)
# rhx: pitch, offset like eta at phi=0
# rhy: roll, offset like eta at phi=90


plot_type = 2 # 0=no plot, 1=matplotlib2D, 2=silx2D, 3=silx3D




#geometry.set_offsets(rhx=0, rhy=0, phi=0, eta=0) #




########################################################################

qx, qy, qz, gint = qconversion.scan_to_qspace_h5(
                                    h5f[sample]["%i.1"%scanno],
                                    cen_pix_hor,
                                    cen_pix_vert,
                                    det_distance,
                                    nbins=bins,
                                    medfilter=False,
                                    geometry=geometry,
                                    monitor=monitor,
                                    roi=roi,
                                    Nav=nav
                                    )



axlabels = ["$Q_%s \\left(\\AA^{-1}\\right)$"%s for s in "xyz"]

if 0: # mayavi not maintained?
    from mayavi import mlab
    QX,QY,QZ = np.mgrid[qx.min():qx.max():1j * nx,
                          qy.min():qy.max():1j * ny,
                          qz.min():qz.max():1j*nz]
    INT = xu.maplog(gint,5.5,0)
    mlab.figure()
    mlab.contour3d(QX, QY, QZ, INT, contours=15, opacity=0.6)
    mlab.colorbar(title="log(int)", orientation="vertical")
    mlab.axes(nb_labels=5, xlabel='Qx', ylabel='Qy', zlabel='Qz')
    mlab.title('SCAN:%i'%scanno)
    mlab.show()




def mycontour(data, numlevels=150, saveto=None, labels=None):
    numcols = len(data)//3
    #erzeuge Grafikfenster (Integriert) Darstellung als Projektion
    plt.figure(figsize=(25,7)) #25*7 zoll
    cbars = []
    for col in range(numcols):
        plt.subplot(1,numcols,col+1,aspect=1) #1 zeile, 3 spalten, wähle spalte 1 an
        x = data[3*col]
        y = data[3*col+1]
        z = data[3*col+2]
        print((col, z.shape))
        img = plt.contourf(x,y,z.T,numlevels) #Darstellung der gesamten Summation
        if not labels is None:
            label = labels[col]
            plt.xlabel(r"Q$_%s$ ($1/\AA$)"%label[0])
            plt.ylabel(r"Q$_%s$ ($1/\AA$)"%label[1])
        plt.axis('tight')
        plt.axis('equal')
        plt.gca().autoscale(tight=True)
        plt.title(sample + "_scan%i"%scanno)
        cbars.append(plt.colorbar(format='%05.2f'))
        cbars[-1].set_norm(Normalize.Normalize(vmin=z.min(),vmax=z.max(),stretch='linear'))
        cbars.append(DraggableColorbar.DraggableColorbar(cbars[-1],img))
        cbars[-1].connect()
    plt.tight_layout()
    if isinstance(saveto, str):
        plt.savefig(saveto)
    plt.show()




if plot_type==1:
    # For example the sums:
    mycontour([qx, qy, xu.maplog(gint.sum(axis=2)),
               qx, qz, xu.maplog(gint.sum(axis=1)),
               qy, qz, xu.maplog(gint.sum(axis=0))],
               #saveto = sample+"_scan%i_sum.png"%scan,
               labels = ["xy", "xz", "yz"]
              )

elif plot_type==2:
    from silx.gui import qt
    from silx.gui.plot.StackView import StackView, StackViewMainWindow
    
    app = qt.QApplication([])
    
    #sv = StackViewMainWindow()# yinverted=True)
    sv = StackView(yinverted=True)
    #sv.setColormap("jet", autoscale=True, normalization="log")
    sv.setColormap("jet", normalization="log", autoscale=False, vmin=gint[gint>0].min(), vmax=gint.max())
    #INT = xu.maplog(gint,5.5,0)
    #plotdata = qsum/histo
    #plotdata[pl.isnan(plotdata)] = 0
    sv.setStack(gint)
    #print(sv.isYAxisInverted())
    sv.setLabels(axlabels)
    sv.setYAxisInverted(True)
    sv.show()
    app.exec_()




elif plot_type==3:
    from silx.gui import qt
    from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
    from silx.gui.plot3d import SFViewParamTree
    
    # Start GUI
    global app  # QApplication must be global to avoid seg fault on quit
    app = qt.QApplication([])

    # Create the viewer main window
    window = ScalarFieldView()
    window.setAttribute(qt.Qt.WA_DeleteOnClose)

    # Create a parameter tree for the scalar field view
    treeView = SFViewParamTree.TreeView(window)
    treeView.setSfView(window)  # Attach the parameter tree to the view

    # Add the parameter tree to the main window in a dock widget
    dock = qt.QDockWidget()
    dock.setWindowTitle('Parameters')
    dock.setWidget(treeView)
    window.addDockWidget(qt.Qt.RightDockWidgetArea, dock)

    # Set ScalarFieldView data
    INT = xu.maplog(gint,5.5,0).transpose(2,1,0)
    #INT = np.log(gint+1)#,5.5,0)
    window.setData(INT)

    # Set scale of the data
    window.setScale(qx[1]-qx[0],
                    qy[1]-qy[0],
                    qz[1]-qz[0])

    # Set offset of the data
    window.setTranslation(qx[0], qy[0], qz[0])

    # Set axes labels
    #window.setAxesLabels(axlabels)
    window.setAxesLabels("Qx","Qy","Qz")

    # Add an iso-surface
    # Add an iso-surface at the given iso-level
    #window.addIsosurface(args.level, '#FF0000FF')
    # Add an iso-surface from a function
    nlevel = 5
    # choose color maps
    #cmap = plt.cm.CMRmap 
    cmap = plt.cm.rainbow
    #cmap = plt.cm.gist_rainbow
    #cmap = plt.cm.Paired
    #cmap = plt.cm.cubehelix
    #cmap = plt.cm.gist_earth
    
    levels = np.linspace(INT.min(), INT.max(), nlevel+2)[1:-1]
    for ii, level in enumerate(levels):
        color = list(cmap(float(ii)/(nlevel-1)))
        color[-1] = (float(ii+1)/(nlevel))**1
        window.addIsosurface(level, color)

    window.show()
    app.exec_()


