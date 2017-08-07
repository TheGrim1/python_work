#!/usr/bin/env python
# taken from pscan_align (Steven Leake)
# reduced by Carsten Richter
# will be extended again later
# wishlist/todo:
# -
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Cursor
import collections



class GenericIndexTracker(object):
    """
        Just a matplotlibe widget that allows scrolling through 
        several images and picking a point.
        No connection to spec.
    """
    _axes_properties = {}
    def __init__(self, ax, data=None, norm="linear", quantum=1.):
        self.ax = ax
        self.fig = ax.figure
        
        ax.format_coord = self.format_coord
        
        if norm=="log":
            self._norm = lambda d: colors.LogNorm(max(d.min(), quantum), d.max())
        elif isinstance(norm, float):
            self._norm = lambda d: colors.PowerNorm(norm, d.min(), d.max())
        else:
            self._norm = lambda d: colors.Normalize(d.min(), d.max())
        
        if data is None:
            if not ax.images:
                raise ValueError("No data found.")
            data = ax.images[-1].get_array()
        self.data = data = np.array(data, ndmin=3)
        
        self.slices = data.shape[0]
        self.ind = 0 # starting picture
        
        if self.slices>1:
            ax.set_title('use scroll wheel to navigate images')
            #self.fig.suptitle('use scroll wheel to navigate images')
        
        
        imkwargs = dict(interpolation="nearest", origin="lower")
        if not ax.images:
            self.im = ax.imshow(data[self.ind], **imkwargs)
            self.cb = plt.colorbar(self.im)
        else:
            self.im = ax.images[-1]
        
        self.cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
        self.fig.canvas.mpl_connect('button_release_event', self.onclick)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.update(True)
    
    def set_extent(self, xmin, xmax, ymin, ymax):
        extent = (xmin, xmax, ymin, ymax)
        self.im.set_extent(extent)
    
    def set_axes_properties(self, **prop):
        """
            This allows to define multiple properties for
            a matplotlib subplot which will be used for the different
            frames when scrolling the mouse wheel.
        """
        for k in prop:
            setter = "set_%s"%k
            val = prop[k]
            if not hasattr(self.ax, setter):
                continue
            if hasattr(val, "__iter__") and len(val)==self.slices:
                self._axes_properties[setter] = val
            else:
                self._axes_properties[setter] = [val]*self.slices
    
    
    def format_coord(self, x, y):
        xlabel = self.ax.xaxis.label._text
        ylabel = self.ax.yaxis.label._text
        ext = self.im._extent
        A = self.im._A
        ix = int((x - ext[0]) / (ext[1] - ext[0]) * A.shape[1])
        iy = int((y - ext[2]) / (ext[3] - ext[2]) * A.shape[0])
        ix = np.clip(ix, 0, A.shape[1]-1)
        iy = np.clip(iy, 0, A.shape[0]-1)
        #print ix, iy
        I = A[iy, ix]
        return '%s=%1.4f, %s=%1.4f, I=%g'%(xlabel, x, ylabel, y, I)
    
    def onscroll(self, event):
        if self.slices==1:
            return
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up': #up should be previous
            ind = np.clip(self.ind - 1, 0, self.slices - 1)
        else:
            ind = np.clip(self.ind + 1, 0, self.slices - 1)
        
        if self.ind != ind:
            self.ind = ind
            self.update(props=True)
        
    def onclick(self,event):
        if not event.inaxes==self.ax:
            print("\nYou did not click on the display.")
            return
        xlabel = self.ax.xaxis.label._text
        ylabel = self.ax.yaxis.label._text
        xdata = event.xdata
        ydata = event.ydata
        print("You selected %s=%.2f, %s=%.2f."%(xlabel, xdata, ylabel, ydata))
        self.POI = xdata, ydata
        #plt.close(self.fig)

    def update(self, props=False):
        if props:
            for k,v in self._axes_properties.items():
                getattr(self.ax, k)(v[self.ind])
        data = self.data[self.ind]
        self.im.set_data(data)
        self.im.set_norm(self._norm(data))
        self.fig.canvas.draw()



ScanRange = collections.namedtuple("ScanRange", ['name', 'start', 'stop', 'numpoints'])


class PScanTracker(GenericIndexTracker):
    """
        This tracker child does the connection to spec
    """
    def __init__(self, ax, pscan_vars, pscan_data, lima_roi, norm="linear",
                       quantum=1., transposed=True):
        if hasattr(pscan_vars, "getValue"):
            pscan_vars = pscan_vars.getValue()
        if hasattr(lima_roi, "getValue"):
            lima_roi = lima_roi.getValue()
        
        # get command
        self.pscan_vars = pscan_vars
        self.pscan_data = pscan_data
        self.command = cmd = pscan_vars["header/cmd"].split()
        
        # pscan motor name, start, stop, numpoints
        m1 = ScanRange(cmd[1], float(cmd[2]), float(cmd[3]), int(cmd[4]))
        m2 = ScanRange(cmd[5], float(cmd[6]), float(cmd[7]), int(cmd[8]))
        
        self.x, self.y = (m2, m1) if transposed else (m1, m2)
        self.transposed = transposed # to look similar to pymca
        
        data = self.load_data()
        
        super(PScanTracker, self).__init__(ax, data, norm, quantum)
        
        self.ax.set_xlabel(self.x.name)
        self.ax.set_ylabel(self.y.name)
        self.set_extent(self.x.start, self.x.stop, self.y.start, self.y.stop)
        
        self.rois = [lima_roi['%i'%(i+1)] for i in range(self.slices)]
        self.set_axes_properties(title=self.rois)
    
    
    def load_data(self):
        data = self.pscan_data.getValue()[1:] # what is the unwanted item 0?
        data = data.reshape((-1, self.x.numpoints, self.y.numpoints, 7))[:,:,2] # what is this 7?? answer: 2 is the sum, there are other things like max.
        if self.transposed:
            data = data.transpose(0, 2, 1)
        return data
        
    
    def reload(self):
        """
            To be called regularly in pscan_live.
        """
        self.data = self.load_data()
        super(PScanTracker, self).update()




