a = np.atleast_1d(np.arange(100))
import numpy as np
import sys, os
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import fileIO.plots.plot_array as pa
import simplecalc.image_align as ia
a = np.atleast_1d(np.arange(100))
b = np.asarray(np.meshgrid(a,a))



aligned_stack, shift = ia.centerofmass_align(rotated)
xth = [(x,i*360/30.0) for i,(x,y) in enumerate(shift)]
xth=np.asarray(xth)
