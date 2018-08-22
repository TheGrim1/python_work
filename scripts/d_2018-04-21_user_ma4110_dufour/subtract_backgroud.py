import sys, os
import numpy as np
sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))

import simplecalc.fitting as fit
import matplotlib.pyplot as plt


def subtract_parabolic_background(frame, bkg_points=None, verbose = False):
    frame_no = np.arange(len(frame))
    data = np.asarray(zip(frame_no,frame))

    if bkg_points != None:
        data = data[bkg_points]

    fitresult = fit.do_cubic_fit(data, verbose=True)
    plt.plot(frame_no,frame,'+b')
    plt.plot(frame_no,fit.cubic_func(fitresult, frame_no),'r-')
    plt.show()
    frame = frame - fit.cubic_func(fitresult, frame_no)
    
    
    return frame
