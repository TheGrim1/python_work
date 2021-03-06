import numpy as np
import math
from math import fabs, atan, sqrt, sin, cos
from scipy import ndimage as ndi

from o5lib.base.o5ccbase import minimaxi, rebin2d as _rebin2d, blowup2d as _blowup2d, thresh_rebin2d, cnv_thresh_rebin2d, hit_process_001


from ilai.seriesproc.genop import                  normalize
from ilai.seriesproc.genop import difference as    sequential_difference
from ilai.seriesproc.genop import closest1D as     closest1d

from o8qq.compute.constants import mathPIH, mathPI, mathE, RPD, DPR

from o8x3.util.utils import make_timer


def cog_var(arr):
    

    arr[19][30]=0 # hotpixel
#    arr[0][764]=0 # hotpixel
#    arr[210][0]=0 # hotpixel

    maxval = np.max((arr))
    if maxval <= 4 :
        res_com = (0,0)
        res_var = 0
    else:
        res_com = ndi.measurements.maximum_position(arr)
        res_com = ndi.measurements.center_of_mass(arr)
        res_var = ndi.variance(arr)
#    print "doing cog, found: \ncog = %s \nvar = %s" % (res_com,res_var)
    return (res_com[0], res_com[1], res_var)
