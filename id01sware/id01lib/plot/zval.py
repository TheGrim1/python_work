import matplotlib.pyplot as plt
import numpy as np
import types

def format_coord(self, x, y):
    xlabel = self.xaxis.label._text
    ylabel = self.yaxis.label._text
    im = self.images[-1]
    ext = im._extent
    A = im._A
    ix = int((x - ext[0]) / (ext[1] - ext[0]) * A.shape[1])
    iy = int((y - ext[2]) / (ext[3] - ext[2]) * A.shape[0])
    ix = np.clip(ix, 0, A.shape[1]-1)
    iy = np.clip(iy, 0, A.shape[0]-1)
    #print ix, iy
    I = A[iy, ix]
    return '%s=%1.4f, %s=%1.4f, I=%g'%(xlabel, x, ylabel, y, I)


def format_axes(ax):
    ax.format_coord = types.MethodType(format_coord, ax)

