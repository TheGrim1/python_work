#!/usr/bin/env python
# taken from pscan_align (Steven Leake)
# reduced by Carsten Richter
# will be extended again later
# wishlist/todo:
#   TODO: add shexacor to specfile inspector, complete spec interface here + test
# -
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Cursor
from matplotlib.patches import Rectangle

import collections

# local
import fileIO.datafiles.save_data as save_data

def run_GenericIndexTracker(inargs):

    print('this is run_GenericIndexTracker ')
    data = inargs[0]
    norm = inargs[1]
    data_fname = inargs[2]
    plt.ioff()
    fig, ax = plt.subplots()

    bla = GenericIndexTracker(ax, data=data, norm=norm, quantum=1.,exit_onclick=False,rectangle_onclick=False, data_fname=data_fname)
    plt.show()

    plt.ion()

    return np.asarray(bla.POI_list)
 
class GenericIndexTracker(object):
    """
        Just a matplotlibe widget that allows scrolling through 
        several images and picking a point.
        No connection to spec.
    """
    _axes_properties = {}
    def __init__(self, ax, data=None, norm="linear", quantum=1.,exit_onclick=False,rectangle_onclick=False, data_fname=None):
        self.ax = ax
        self.fig = ax.figure
        ax.format_coord = self.format_coord

        if norm=="log":
            self._norm = lambda d: colors.LogNorm(d[d>0].min(), d.max())
        elif isinstance(norm, float):
            self._norm = lambda d: colors.PowerNorm(norm, d[d>0].min(), d.max())
        else:
            self._norm = lambda d: colors.Normalize(d.min(), d.max())

        if data is None:
            if not ax.images:
                raise ValueError("No data found.")
            data = ax.images[-1].get_array()
        self.data = data = np.array(data, ndmin=3)
        self.slices = data.shape[0]
        self.ind = 0 # starting picture
        self.data_fname = data_fname
        
        if self.slices>1:
            ax.set_title('frame {}'.format(self.ind))
            #self.fig.suptitle('use scroll wheel to navigate images')


        imkwargs = dict(interpolation="nearest",
                        origin="lower",
                        norm=self._norm(data[self.ind]))
        if not ax.images:
            self.im = ax.matshow(data[self.ind], **imkwargs)
            self.cb = plt.colorbar(self.im)
        else:
            self.im = ax.images[-1]

        if not hasattr(self, "cursor"): # first time
            self.cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
            if not rectangle_onclick:
                self.fig.canvas.mpl_connect('button_release_event', self.onclick)

        self.update(True)

        self.exit_onclick = exit_onclick
        self.rectangle_onclick = rectangle_onclick

        if self.rectangle_onclick and not self.exit_onclick:
            self.rect = Rectangle((0,0), 1, 1, facecolor='None', edgecolor='green')
            self.x0 = 0
            self.y0 = 0
            self.x1 = 0
            self.y1 = 0
            self.ax.add_patch(self.rect)
            self.toggle = False
            self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
            self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
            self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.POI_list=[]


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
        I = A[iy, ix]
        return '%s=%1.4f, %s=%1.4f, I=%g'%(xlabel, x, ylabel, y, I)


    def onclick(self,event):
        if not event.inaxes==self.ax:
            print("\nYou did not click on the display.")
            return
        xlabel = self.ax.xaxis.label._text
        ylabel = self.ax.yaxis.label._text
        xdata = event.xdata
        ydata = event.ydata
        print("You selected:    %s %.2f    %s %.2f"%(xlabel, xdata, ylabel, ydata))
    
        self.POI = xdata, ydata
        self.POI_mot_nm = xlabel, ylabel
        self.POI_list.append((xdata, ydata))
        #self.ax.annotate("p%i"%len(self.POI_list), xy=(ydata, xdata),)

        print('frame {} of {}'.format(self.ind+1,self.slices))
        
        if self.ind+1 == self.slices:
            self.end()

            
            return
        #print("%s %s" % (event.button, event.step))
        ind = np.clip(self.ind + 1, 0, self.slices - 1)

        if self.ind != ind:
            self.ind = ind
            self.update(props=True)
            
    def end(self):
        plt.close(self.fig)

        fname = self.data_fname
        if fname != None:
           
            data = np.asarray(self.POI_list)
            save_data.save_data(fname=fname, data=data, header = ['pxl_x','pxl_y'])
            print('data saved in {}'.format(fname))
    
    def on_press(self,event):
        self.x0 = event.xdata
        self.y0 = event.ydata    
        self.x1 = event.xdata
        self.y1 = event.ydata
        try:    
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
        except TypeError:
            print "You clicked outside the window - try again"

        self.rect.set_linestyle('dashed')
        self.ax.figure.canvas.draw()
        self.toggle = True

    def on_motion(self,event):
        if self.on_press is True:
            return
        if self.toggle:
            self.x1 = event.xdata
            self.y1 = event.ydata
            try:
                self.rect.set_width(self.x1 - self.x0)
                self.rect.set_height(self.y1 - self.y0)
                self.rect.set_xy((self.x0, self.y0))
            except TypeError:
                print "You moved the mouse outside the window - try again"
            self.rect.set_linestyle('dashed')
            self.ax.figure.canvas.draw()

    def on_release(self, event):
        if not event.inaxes==self.ax:
            print("\nYou did not click on the display.")
            self.on_press=False
            self.toggle=False
            return  
        #print 'release'
        self.x1 = event.xdata
        self.y1 = event.ydata
        try:
            self.rect.set_width(self.x1 - self.x0)
            self.rect.set_height(self.y1 - self.y0)
            self.rect.set_xy((self.x0, self.y0))
        except TypeError:
            print "You clicked outside the window - try again"
        self.rect.set_linestyle('solid')
        self.ax.figure.canvas.draw()
        plt.close(self.fig) 

    def update(self, props=False):
        if props:
            for k,v in self._axes_properties.items():
                getattr(self.ax, k)(v[self.ind])
        data = self.data[self.ind]
        self.ax.set_title('frame {}'.format(self.ind))
        self.im.set_data(data)
        self.im.set_norm(self._norm(data))
        self.fig.canvas.draw()




