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
from silx.io.specfile import SpecFile


import collections

from . import SpecClientWrapper
from . import image


class GenericIndexTracker(object):
    """
        Just a matplotlibe widget that allows scrolling through 
        several images and picking a point.
        No connection to spec.
    """
    _axes_properties = {}
    def __init__(self, ax, data=None, norm="linear", quantum=1.,exit_onclick=False,rectangle_onclick=False):
        self.ax = ax
        self.fig = ax.figure
        ax.format_coord = self.format_coord

        if norm=="log":
            self._norm = lambda d: colors.LogNorm(d[d>0].min(), d.max())
        elif isinstance(norm, float):
            self._norm = lambda d: colors.PowerNorm(norm, d[d>0].min(), d.max())
        else:
            self._norm = lambda d: colors.Normalize(d[d>0].min(), d.max())

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


        imkwargs = dict(interpolation="nearest",
                        origin="lower",
                        norm=self._norm(data[self.ind]))
        if not ax.images:
            self.im = ax.imshow(data[self.ind], **imkwargs)
            self.cb = plt.colorbar(self.im)
        else:
            self.im = ax.images[-1]

        if not hasattr(self, "cursor"): # first time
            print('switch on onclick',rectangle_onclick)
            self.cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
            if not rectangle_onclick:
                self.fig.canvas.mpl_connect('button_release_event', self.onclick)
                print('switch on onclick')
            self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
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
        print("You selected:    %s %.2f    %s %.2f"%(xlabel, xdata, ylabel, ydata))
        self.POI = xdata, ydata
        self.POI_mot_nm = xlabel, ylabel
        self.POI_list.append((xdata, ydata))
        #self.ax.annotate("p%i"%len(self.POI_list), xy=(ydata, xdata),)
        if self.exit_onclick:
            plt.close(self.fig)

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
        self.im.set_data(data)
        self.im.set_norm(self._norm(data))
        self.fig.canvas.draw()

class GenericIndexTracker1D(object):
    """
        Just a matplotlibe widget that allows scrolling through 
        several images and picking a point.
        No connection to spec.
    """
    _axes_properties = {}
    def __init__(self, ax, data=None, norm="linear", quantum=1.,exit_onclick=False,rectangle_onclick=False):
        self.ax = ax
        self.fig = ax.figure
        #ax.format_coord = self.format_coord

        if norm=="log":
            self._norm = lambda d: colors.LogNorm(d[d>0].min(), d.max())
        elif isinstance(norm, float):
            self._norm = lambda d: colors.PowerNorm(norm, d[d>0].min(), d.max())
        else:
            self._norm = lambda d: colors.Normalize(d[d>0].min(), d.max())

        if data is None:
            if not ax.lines:
                raise ValueError("No data found.")
            data = ax.lines[-1].get_array()
        self.data = data = np.array(data, ndmin=3)
        self.slices = data.shape[0]
			
        self.ind = 0 # starting picture

        if self.slices>1:
            ax.set_title('use scroll wheel to navigate images')
            #self.fig.suptitle('use scroll wheel to navigate images')
        

        imkwargs = dict(interpolation="nearest",
                        origin="lower",
                        norm=self._norm(data[self.ind]))
        if not ax.lines:
            self.ln = ax.plot(data[self.ind,:,0])
            #self.cb = plt.colorbar(self.im)
        else:
            self.ln = ax.lines[-1]

        if not hasattr(self, "cursor"): # first time
            self.cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
            self.fig.canvas.mpl_connect('button_release_event', self.onclick)
            self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        self.update(True)

        self.exit_onclick = exit_onclick

        self.POI_list=[]


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
        print("You selected:    %s %.2f  "%(xlabel, xdata))
        self.POI = xdata
        self.POI_mot_nm = xlabel
        self.POI_list.append((xdata, ydata))
        #self.ax.annotate("p%i"%len(self.POI_list), xy=(ydata, xdata),)
        if self.exit_onclick:
            plt.close(self.fig)


    def update(self, props=False):
        if props:
            for k,v in self._axes_properties.items():
                getattr(self.ax, k)(v[self.ind])
        data = self.data[self.ind,:,0]
        self.ln[0].set_ydata(data)
        self.ax.set_ylim(min(data),max(data))
        #self.ln.set_norm(self._norm(data))
        self.fig.canvas.draw()

ScanRange = collections.namedtuple("ScanRange", ['name', 'start', 'stop', 'numpoints'])

class PScanTracker(GenericIndexTracker):
    """
        This tracker child does the connection to spec
    """
    def __init__(self, ax, specclient, norm="linear",
                       quantum=1., transposed=True, exit_onclick=False, rectangle_onclick=False):
        """
            Class to fill a matplotlib axes with with data
            from fast spec scans (`pscan`)

            Inputs:
                ax : matplotlib.Axes
                    axes used to show data

                specclient : SpecClientWrapper.SpecClientSession
                    a wrapped SpecClient instance to handle
                    communication with spec

                norm : str, float
                    normalization of the colormap

        """
        if not isinstance(specclient, SpecClientWrapper.SpecClientSession):
            raise ValueError("Need `SpecClientWrapper.SpecClientSession` "
                             "instance as second argument.")

        # get command
        self.pscan_vars = pscan_vars = specclient.get_sv("PSCAN_ARR")
        self.specclient = specclient
        self.command = cmd = pscan_vars["header/cmd"].split()
        print(cmd)

        lima_roi, device = specclient.find_roi_list()

        # pscan motor name, start, stop, numpoints
        m1 = ScanRange(cmd[1], float(cmd[2]), float(cmd[3]), int(cmd[4]))
        m2 = ScanRange(cmd[5], float(cmd[6]), float(cmd[7]), int(cmd[8]))

        self.x, self.y = (m2, m1) if transposed else (m1, m2)
        self.transposed = transposed # to look similar to pymca

        data = self.load_data()

        self._args = norm, quantum, transposed
        super(PScanTracker, self).__init__(ax, data, norm, quantum, exit_onclick)

        self.ax.set_xlabel(self.x.name)
        self.ax.set_ylabel(self.y.name)
        self.set_extent(self.x.start, self.x.stop, self.y.start, self.y.stop)

        self.rois = lima_roi #['%i'%(i+1)] for i in range(self.slices)]
        self.set_axes_properties(title=self.rois)

        basename = os.path.basename(pscan_vars["file"])
        scan_no = pscan_vars["scan_no"]
        self.start_time = start_time = pscan_vars["timestamp/_pscan_doscan01"]
        title = "File: %s; Scan: %s; %s"%(basename, scan_no, start_time)
        if hasattr(self, "figtitle"):
            self.figtitle.set_text(title)
        else:
            self.figtitle = self.fig.suptitle(title)

    
    def load_data(self):
        data = self.specclient.get_sv("PSCAN_ROICOUNTER_DATA")[1:]
        try:
            data = data.reshape((-1, self.x.numpoints, self.y.numpoints, 7))[:,:,:,2] # what is this 7?? answer: 2 is the sum, there are other things like max.
        except ValueError:
            print("Warning: reshape failed.")
            return self.data
        if self.transposed:
            data = data.transpose(0, 2, 1)
        if not (data>0).any():
            return self.data
        return data

    def re_init(self):
        return self.__init__(self.ax, self.specclient, *self._args)

    def reload(self):
        """
            To be called regularly in pscan_live.
        """
        # check if new scan was started:
        pscan_vars = pscan_vars = self.specclient.get_sv("PSCAN_ARR")
        start_time = pscan_vars["timestamp/_pscan_doscan01"]
        if start_time!=self.start_time:
            #print(scan_no,self.scan_no,basename,self.basename)
            self.specclient.varcache.pop("PSCAN_ARR")
            self.re_init()
        else:
            self.data = self.load_data()
            super(PScanTracker, self).update()

class Annotate(GenericIndexTracker):
    """
        This tracker child does the connection to spec
    """
    def __init__(self, ax, specclient, norm="linear",
                       quantum=1., transposed=True, exit_onclick=False, rectangle_onclick=False):
        """
            Class to fill a matplotlib axes with with data
            from fast spec scans (`pscan`)

            Inputs:
                ax : matplotlib.Axes
                    axes used to show data

                specclient : SpecClientWrapper.SpecClientSession
                    a wrapped SpecClient instance to handle
                    communication with spec

                norm : str, float
                    normalization of the colormap

        """
        if not isinstance(specclient, SpecClientWrapper.SpecClientSession):
            raise ValueError("Need `SpecClientWrapper.SpecClientSession` "
                             "instance as second argument.")
        # get command
        #self.specclient = specclient
        #lima_roi, device = specclient.find_roi_list()

        #print("###################\nDevices available: \n")
        #print(device)
        #print("###################\n")

#        self.device = device  # not robust against multiple devices!


#        data = self.load_data()

        #self._args = norm, quantum, transposed
#        super(Annotate, self).__init__(ax, data, norm, quantum, 
#                                       exit_onclick, rectangle_onclick)
        super(Annotate, self).__init__(ax, norm, quantum, 
                                       exit_onclick, rectangle_onclick)
#       self.ax.invert_yaxis()


#    def load_data(self):
#        data = self.specclient.get_last_image(self.device)[None,:,:]
#        return data

class PScanTracker1D(GenericIndexTracker1D):
    """
        This tracker child does the connection to spec
    """
    def __init__(self, ax, specclient, norm="linear",
                       quantum=1., transposed=True, exit_onclick=False, rectangle_onclick=False):
        """
            Class to fill a matplotlib axes with with data
            from fast spec scans (`pscan`)

            Inputs:
                ax : matplotlib.Axes
                    axes used to show data

                specclient : SpecClientWrapper.SpecClientSession
                    a wrapped SpecClient instance to handle
                    communication with spec

                norm : str, float
                    normalization of the colormap

        """
        if not isinstance(specclient, SpecClientWrapper.SpecClientSession):
            raise ValueError("Need `SpecClientWrapper.SpecClientSession` "
                             "instance as second argument.")

        # get command
        self.pscan_vars = pscan_vars = specclient.get_sv("PSCAN_ARR")
        self.specclient = specclient
        self.command = cmd = pscan_vars["header/cmd"].split()
        print(cmd)

        lima_roi, device = specclient.find_roi_list()

        # pscan motor name, start, stop, numpoints
        m1 = ScanRange(cmd[1], float(cmd[2]), float(cmd[3]), int(cmd[4]))
        m2 = ScanRange(cmd[5], float(cmd[6]), float(cmd[7]), int(cmd[8]))

        self.x, self.y = (m2, m1) if transposed else (m1, m2)
        self.transposed = transposed # to look similar to pymca

        data = self.load_data()

        self._args = norm, quantum, transposed
        super(PScanTracker1D, self).__init__(ax, data, norm, quantum, exit_onclick)
        
        self.ln[0].set_xdata(np.arange(float(cmd[2]),float(cmd[3]),(float(cmd[3])-float(cmd[2]))/int(cmd[4])))
        self.ax.set_xlim(float(cmd[2]),float(cmd[3]))
        self.ax.set_xlabel(self.y.name)
        self.ax.set_ylabel("Intensity")
        self.rois = lima_roi #['%i'%(i+1)] for i in range(self.slices)]
        self.set_axes_properties(title=self.rois)

        basename = os.path.basename(pscan_vars["file"])
        scan_no = pscan_vars["scan_no"]
        self.start_time = start_time = pscan_vars["timestamp/_pscan_doscan01"]
        title = "File: %s; Scan: %s; %s"%(basename, scan_no, start_time)
        if hasattr(self, "figtitle"):
            self.figtitle.set_text(title)
        else:
            self.figtitle = self.fig.suptitle(title)


    
    def load_data(self):
        data = self.specclient.get_sv("PSCAN_ROICOUNTER_DATA")[1:]
        try:
            data = data.reshape((-1, self.x.numpoints, self.y.numpoints, 7))[:,:,:,2] # what is this 7?? answer: 2 is the sum, there are other things like max.
        except ValueError:
            print("Warning: reshape failed.")
            return self.data
        if self.transposed:
            data = data.transpose(0, 2, 1)
        if not (data>0).any():
            return self.data
        return data

    def re_init(self):
        return self.__init__(self.ax, self.specclient, *self._args)

    def reload(self):
        """
            To be called regularly in pscan_live.
        """
        # check if new scan was started:
        pscan_vars = pscan_vars = self.specclient.get_sv("PSCAN_ARR")
        start_time = pscan_vars["timestamp/_pscan_doscan01"]
        if start_time!=self.start_time:
            #print(scan_no,self.scan_no,basename,self.basename)
            self.specclient.varcache.pop("PSCAN_ARR")
            self.re_init()
        else:
            self.data = self.load_data()
            super(PScanTracker1D, self).update()

class PScanTrackerSpecfile(GenericIndexTracker):
    """
        This tracker child does the connection to spec
    """
    def __init__(self, ax, spec_fn, scan_no, norm="linear",
                       quantum=1., transposed=True, exit_onclick=False, rectangle_onclick=False):
		
		self.spec_fn = spec_fn
		self.scan_no = scan_no
		
		scans = SpecFile(self.spec_fn)
		try: 
			index=scans.keys().index(self.scan_no)
		except:
			print self.scan_no, '  not in specfile: ' , self.spec_fn
		
		self.scandata=scans[index]
		
		# finding the counters of interest
		#roiIndexstart = data.header.index('#C --------------------- ROIS')+1
		#for i,line in enumerate(data.header):
		#	if line.startswith('#C image'):
		#		roiIndexEnd=i
		#		
		#rois = data.header[roiIndexstart:roiIndexEnd]
		#roi_list=[]
		#[roi_list.append(roi.split()[1]) for roi in rois]
		#new_roi_list=[]
		#[new_roi_list.append(roi) for roi in roi_list if data.labels.count(roi)>0]
		#roi_list=new_roi_list
		tmp_list=[]
		
		# better way of finding the counters of interest		
		
		for label in self.scandata.labels:
			if not label in ['timer','imgnr','adcX','adcY','adcZ','adc3']:
				tmp_list.append(label)
				
		lima_roi=tmp_list
		
		scan_str=''
		
		for i in self.scandata.header: 
			if i.startswith('#S '):
				scan_str=i.split()[2:]
		
		m1 = ScanRange(scan_str[1], float(scan_str[2]), float(scan_str[3]), int(scan_str[4]))
		m2 = ScanRange(scan_str[5], float(scan_str[6]), float(scan_str[7]), int(scan_str[8]))		

		self.x, self.y = (m2, m1) if transposed else (m1, m2)
		self.transposed = transposed # to look similar to pymca
		
		motor_dict=['thx','thy','thz','pix','piy','piz','eta','phi','del','nu']
		motor_dict.remove(m1.name)
		motor_dict.remove(m2.name)
		send2spec=['groupdel inspect_specfile']
		specstr='groupadd inspect_specfile'
		specstr1='groupaddpos inspect_specfile p0'
		for motor in motor_dict:
			specstr+=' '+motor
			specstr1+=' %.3f'%self.scandata.motor_position_by_name(motor)
		# ad shexacor2beam values - could even move to a lab frame description?
		print(specstr)
		print(specstr1)
		
		pscan_data = self.scandata.data.copy().reshape((len(self.scandata.labels),int(self.x.numpoints),int(self.y.numpoints)))
		pscan_data_tmp = np.zeros((len(lima_roi),int(self.x.numpoints),int(self.y.numpoints)))
		for ii, key in enumerate(lima_roi):
			print ii,key, self.scandata.labels.index(key)
			pscan_data_tmp[ii,:,:]=pscan_data[self.scandata.labels.index(key),:,:]

		data = pscan_data_tmp
		self._args = norm, quantum, transposed
		super(PScanTrackerSpecfile, self).__init__(ax, data, norm, quantum,exit_onclick,rectangle_onclick)
		self.ax.set_xlabel(self.x.name)
		self.ax.set_ylabel(self.y.name)
		self.set_extent(self.x.start, self.x.stop, self.y.start, self.y.stop)
		self.rois = lima_roi #['%i'%(i+1)] for i in range(self.slices)]
		self.set_axes_properties(title=self.rois)

		basename = self.spec_fn
		scan_no = self.scan_no
		self.start_time = start_time = self.scandata.scan_header_dict['D']

		title = "File: %s; Scan: %s; %s"%(basename, scan_no, start_time)
		if hasattr(self, "figtitle"):
			self.figtitle.set_text(title)
		else:
			self.figtitle = self.fig.suptitle(title)



