#!/usr/bin/env python3
#----------------------------------------------------------------------
# Description:
# Author: Carsten Richter <carsten.richter@esrf.fr>
# Created at: Sa 6. Mai 16:04:24 CEST 2017
# Computer: lid01gpu1.
# System: Linux 3.16.0-4-amd64 on x86_64
#----------------------------------------------------------------------
#
# App for reading out subsequent camera images from given Url
# - image feature matching
# - determine shifts / rotations
# - move center of rotation to point of interest
#
# Currently relies on the newest SILX version (0.5.0)
# Can work for python 3 and python 2
# A suitable python 3 environment is here:
#
#   source /data/id01/inhouse/crichter/venv3.4/bin/activate
#----------------------------------------------------------------------
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import os

os.environ.pop("http_proxy", None) # for ID01
os.environ.pop("https_proxy", None)

import sys

for p_i in range(len(sys.path)):
    if "dist-packages" in sys.path[p_i]:
        sys.path.append(sys.path.pop(p_i))


import time
import numpy as np
print("Using numpy %s"%np.__version__)
from scipy import linalg, ndimage
from PIL import Image
print("Using PIL.Image %s"%Image.VERSION)
from PyQt4 import QtGui as Q
from PyQt4 import QtCore
print("Using PyQt %s"%QtCore.QT_VERSION_STR)

import platform
PV = platform.python_version()
_use_console = True
_use_console = _use_console and PV.startswith("3.")

import silx
print("Using silx %s"%silx.version)
from silx.gui import plot
from silx.gui.plot import PlotActions
import silx.gui.icons
from silx.image import sift
if _use_console:
    from silx.gui import console

print('using local id01lib')
sys.path.append('/data/id13/inhouse2/AJ/skript/id01sware')
import id01lib
from id01lib import image
iconpath = os.path.dirname(os.path.join(id01lib.__file__))
iconpath = os.path.join(iconpath, "media", "camview.png")


#_default_url = "http://220.221.164.165:8000/jpg/image.jpg"
#_default_url = "http://skycam.mmto.arizona.edu/skycam/latest_image.png"
#_default_url = "http://jimstar11.com/DSICam/SkyEye.jpg"
#_default_url = "http://www.webcam.cannstatter-volksfest.de/2013/live/live.jpg"
_default_url = "http://vidid011.esrf.fr/jpg/1/image.jpg"

_default_motors = ["thx", "thy"]

_hints = dict()
_hints["calibration"] = ('Move motor %i, take picture and press to get '
                         'calibration in pixel per motor step.')
_hints["Exposure"] = 'Press to acquire new picture.'
_hints["Get COR"] = ('Identifies features on both images, estimates '
                     'the affine transform between them and returns '
                     'Center Of Rotation.')
_hints["POI to COR"] = ('Move selected Point Of Interest into Center Of '
                      'Rotation')
_hints["Get Sharpness"] = ('Compute a measure for the sharpness of the image '
                        'in arbitrary units. Uses the latest image.')
_hints["selectPOI"] = 'Select Point of Interest (POI)'
_hints["Navg"] = 'Number of subsequent camera images to average'
_hints["enhance"] = 'Strech contrast of the camera image'
_hints["saveit"] = 'Save the new image to the current directory'



class CrosshairAction(PlotActions.CrosshairAction):
    """
        Overridden silx class
    """
    def _actionTriggered(self, checked=False):
        super(CrosshairAction, self)._actionTriggered(checked)
        if checked:
            self.plot.setInteractiveMode("select")
        else:
            self.plot.setInteractiveMode(**self.plot._defaultMode)



class ClearPointsAction(PlotActions.PlotAction):
    def __init__(self, plot, parent=None):
        super(ClearPointsAction, self).__init__(
                            plot,
                            icon='image-select-erase',
                            text='Clear SIFT keypoints',
                            tooltip='Clear keypoints found by SIFT',
                            triggered=self.trigger,
                            parent=parent)

    def trigger(self):
        self.plot.update_keypoints(None)



class CamPlot(plot.PlotWindow):
    roi = None
    poi = None
    def __init__(self, data=None, title=None, parent=None):
        super(CamPlot, self).__init__(parent=parent, resetzoom=True, 
                             autoScale=False,
                             logScale=False, grid=False,
                             curveStyle=False, colormap=True,
                             aspectRatio=True, yInverted=False,
                             copy=True, save=True, print_=True,
                             control=False,
                             roi=False, mask=False)
        self.setXAxisAutoScale(True)
        self.setYAxisAutoScale(True)
        self.setKeepDataAspectRatio(True)
        self.setYAxisInverted(True)
        #self.setKeepDataAspectRatio(True)
        
        if not data is None:
            self.addImage(data, resetzoom=True)
        self.setGraphTitle(title)
        
        
        clearpoints = ClearPointsAction(self)
        self.toolBar().addAction(clearpoints)
        self._clearpoints = clearpoints
        
        
        #r = self.getDataRange()
        #self.setGraphXLimits(r[0][0], r[0][1])
        #self.setGraphYLimits(r[1][0], r[1][1])
        #self.resetZoom()
        #self.profile = plot.Profile.ProfileToolBar(plot=self)
        #self.addToolBar(self.profile)
    
    def update_roi(self, event):
#        if "button" in event and event["button"] == "right":
#            self.remove("roi")
#            self.roi = None
#            return # Problem: No right click signal in draw mode
        
        xlim = np.clip(event["xdata"], 0, None)
        ylim = np.clip(event["ydata"], 0, None)
        xlim.sort()
        ylim.sort()
        
        if xlim[0]==xlim[1] or ylim[0]==ylim[1]:
            self.remove("roi")
            self.roi = None
            if self.getInteractiveMode()["mode"] is 'draw':
                self.parent().parent().echo("Empty ROI -> removed ROI.")
            
            return
        
        self.roi = xlim.astype(int), ylim.astype(int)
        
        
        x = [xlim[i] for i in (0,1,1,0,0)]
        y = [ylim[i] for i in (0,0,1,1,0)]
        self.addCurve(x, y, resetzoom=False, legend="roi", color="r")
    
    
    def update_poi(self, event):
        self.poi = poi = event["x"], event["y"]
        m = self.addMarker(poi[0], poi[1], symbol="o",
                      legend="poi", color=(.3,1.,1.,1.), text="POI")
        #m = self._getItem("marker", m)
        #print(m.getSymbolSize())
        #m.setSymbolSize(1)
        #print(m.getSymbolSize())
        #self.addCurve([event["x"]], [event["y"]], symbol="o", linestyle=" ",
        #              legend="poi", linewidth=5, color="c", resetzoom=False)
        #c = self.getCurve("poi")
        #c.setSymbolSize(10)
    
    
    def get_roi_data(self):
        imdata= self.getImage().getData()
        if not self.roi is None:
            xlim, ylim = self.roi
            roidata = imdata[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            return roidata
        else:
            return imdata
    
    def update_keypoints(self, xy=None):
        plotcfg = dict(legend="keypoints", color=(.3,1.,.3,.8), symbol=".",
                       resetzoom=False, linestyle=" ")
        if xy is None:
            self.remove(plotcfg['legend'])
        else:
            self.addCurve(xy[0], xy[1], **plotcfg)




class ControlWidget(Q.QWidget):
    Input = dict()
    def __init__(self, parent=None, **kw):
        super(ControlWidget, self).__init__(parent=parent, **kw)
        self.home()
    
    def home(self):
        font = Q.QFont()
        font.setPointSize(9)
        self.setFont(font)
        
        layout = Q.QHBoxLayout(self)
        self.splitter = splitter = Q.QSplitter(QtCore.Qt.Horizontal) 
        layout.addWidget(splitter)
        _reg = self.registerWidget
        _onlyint = Q.QIntValidator()
        self.form = form = Q.QFrame(self)
        form.setFrameShape(Q.QFrame.StyledPanel)
        form.layout = Q.QFormLayout(form)

        hbox = Q.QHBoxLayout()
        url = _reg(Q.QLineEdit(_default_url), "url")
        enhance = Q.QCheckBox('enhance', self)
        enhance.setStatusTip(_hints['enhance'])
        enhance = _reg(enhance, "enhanced")

        saveit = Q.QCheckBox('save', self)
        saveit.setStatusTip(_hints['saveit'])
        saveit = _reg(saveit, "saveit")

        
        Navg = Q.QLineEdit("1")
        Navg.setValidator(_onlyint)
        Navg.setStatusTip(_hints['Navg'])
        Navg.setMaxLength(3)
        Navg.setFixedWidth(25)
        Navg = _reg(Navg, "Navg")
        
        hbox.addWidget(Q.QLabel("URL"))
        hbox.addWidget(url)
        hbox.addSpacing(5)
        hbox.addWidget(Q.QLabel("Navg"))
        hbox.addWidget(Navg)
        hbox.addSpacing(5)
        hbox.addWidget(enhance)
        hbox.addWidget(saveit)
        form.layout.addRow(hbox)

        hbox = Q.QHBoxLayout()
        for k in ("E&xposure", "Get CO&R", "POI to COR", "Get Sharpness"):
            name = k.replace("&","")
            btn = _reg(Q.QPushButton(k, self), name)
            btn.setStatusTip(_hints[name])
            hbox.addWidget(btn)
        form.layout.addRow(hbox)
        
        
        
        # Calibration:
        for i in range(1,3):
            motor = _default_motors[i-1]
            hbox = Q.QHBoxLayout()
            MBtn = _reg(Q.QLineEdit(motor), "Mot%i"%i)
            MBtn.setMinimumWidth(40)
            hbox.addWidget(MBtn)
            hbox.addWidget(Q.QLabel("Step"))
            hbox.addWidget(_reg(Q.QLineEdit("0.05"), "Step%i"%i))
            CBtn = Q.QPushButton("Calib. Mot. #%i"%i, self)
            CBtn.setStatusTip(_hints["calibration"]%i)
            hbox.addWidget(_reg(CBtn, "Cal%i"%i))
            hbox.addWidget(_reg(Q.QLineEdit("#####"), "CalRes_%i"%i))
            form.layout.addRow(Q.QLabel("Motor #%i"%i), hbox)
        
        
        [self.Input["CalRes_%i"%i].setMinimumWidth(100) for i in (1,2)]
        
        form.layout.addRow(_reg(Q.QTextEdit(""), "output"))
        self.Input["output"].setReadOnly(True)
        textFont = Q.QFont("Monospace", 9)
        textFont.setStyleHint(Q.QFont.Monospace)
        self.Input["output"].setCurrentFont(textFont)
        
        
        splitter.addWidget(form)
        
        if _use_console:
            banner = "Inspect/Modify `MainWindow` App instance."
            ipython = console.IPythonWidget(self)#, custom_banner=banner)
            #ipython.banner += banner
            mainWindow = self.parent().parent()
            
            ipython.pushVariables({"MainWindow": mainWindow})
            #ipython.font_size = 
            ipython.change_font_size(-2)
            self.console = ipython
            #ipython.clear()
            splitter.addWidget(ipython)
        else:
            splitter.addWidget(Q.QPushButton("Dummy"))
        
        splitter.setSizes([500,500])
        
        self.setLayout(layout)
    
    def registerWidget(self, QtObject, name):
        self.Input[name] = QtObject
        if isinstance(QtObject, Q.QLineEdit):
            #QtObject.setFixedWidth(length)
            pass
        else:
            QtObject.resize(QtObject.minimumSizeHint())
        return QtObject
        
    



class Window(Q.QMainWindow):
    _ignoreEvent = False
    _eventSource = None
    resultsCOR = dict()
    calibration = dict()
    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(200, 100, 1000, 750)
        self.setWindowTitle("Cam view processing")
        if iconpath is not None and os.path.isfile(iconpath):
            print("setting icon %s"%iconpath)
            self.setWindowIcon(Q.QIcon(iconpath))
        
        extractAction = Q.QAction("&Quit", self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip('Leave The App')
        extractAction.triggered.connect(self.close_application)

        #self.setStatusBar(Q.QStatusBar())
        self.statusBar()
        
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(extractAction)
        
        #extractAction = Q.QAction(Q.QIcon('todachoppa.png'), 'Quit', self)
        extractAction = Q.QAction('Quit', self)
        extractAction.triggered.connect(self.close_application)
        extractAction.setStatusTip('Leave The App')
        self.toolBar = self.addToolBar("Extraction")
        self.toolBar.addAction(extractAction)
        
        self.home()
        #self.show()

    def home(self):
        cw = Q.QWidget(self)
        self.grid = g = Q.QGridLayout(cw)
        self.setCentralWidget(cw)
        
        
        data = np.random.random((512,512))
        
        self.plotLeft  = pleft  = CamPlot(data, "Latest", cw)
        self.plotRight = pright = CamPlot(data, "Previous", cw)
        g.addWidget(pleft,  0,0)
        g.addWidget(pright, 0,1)
        
        crosshair = CrosshairAction(pleft, color="b")
        crosshair.setToolTip(_hints["selectPOI"])
        pleft.toolBar().addAction(crosshair)
        pleft.crosshair = crosshair
        
        pleft.setCallback( lambda event: self.handle_event(event, "l"))
        pright.setCallback(lambda event: self.handle_event(event, "r"))
        
        pleft.setInteractiveMode("draw", shape="rectangle"
                                       , color=(1.,1.,0.,0.8))
        
        for p in (pleft, pright):
            p._defaultMode = p.getInteractiveMode()
        
        self.control = control = ControlWidget(cw)
        g.addWidget(control, 1, 0, 1, 2)
        # control.adjustSize()
        
        # Connect:
        control.Input["Exposure"].clicked.connect(self.update_plots)
        control.Input["Get COR"].clicked.connect(self.get_center_of_rotation)
        control.Input["POI to COR"].clicked.connect(self.poi_to_cor)
        control.Input["Get Sharpness"].clicked.connect(self.calc_sharpness)
        control.Input["Cal1"].clicked.connect(lambda: self.calibrate(1))
        control.Input["Cal2"].clicked.connect(lambda: self.calibrate(2))
        
        #self.plotLeft.resetZoom() #doesn't work
        self.show()


    def handle_event(self, event, side):
        if event["event"] is "drawingFinished":
            #print(event["xdata"], event["ydata"])
            for p in [self.plotLeft, self.plotRight]:
                p.update_roi(event)
            
        elif event["event"]=="limitsChanged"  \
          and not self._eventSource==event["source"]  \
          and not self._ignoreEvent:
            self._eventSource=event["source"]
            self._ignoreEvent = True
            if side is "l":
                self.plotRight.setLimits(*(event["xdata"]+event["ydata"]))
            elif side is "r":
                self.plotLeft.setLimits(*(event["xdata"]+event["ydata"]))
            self._ignoreEvent = False
        elif event["event"] is "mouseClicked" and side is "l":
            if event["button"] is "left" and \
              not self.plotLeft.getGraphCursor() is None:
                for p in [self.plotLeft, self.plotRight]:
                    p.update_poi(event)
#            if event["button"] is "right":
#                for p in [self.plotLeft, self.plotRight]:
#                    p.update_roi(event)

    def update_plots(self):
        iso_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        url = self.control.Input["url"].text()#.toAscii()
        url = str(url)
        Navg = int(self.control.Input["Navg"].text())#.toAscii()
        if Navg < 1:
            self.echo("Error: need Navg > 0")
            return
        Ntext = "once" if Navg is 1 else "%i times"%Navg
        self.echo("Exposure - %s"%iso_time)
        self.echo("Fetching %s %s..."%(url, Ntext))
        try:
            img = image.url2array(url, Navg)
            self.echo("Image shape: (%i, %i)"%img.shape)
        except Exception as emsg:
            self.echo("...failed: %s"%emsg)
            return
        
        do_enhance = self.control.Input["enhanced"].checkState()
        if do_enhance:
            img = image.stretch_contrast(img)

        do_save = self.control.Input["saveit"].checkState()
        if do_save:
            try:
                impath = "CamView_%s.png"%iso_time
                im = Image.fromarray(img*255./img.max())
                im = im.convert("RGB")
                im.save(impath)
                self.echo("Saved image to: %s"%os.path.abspath(impath))
            except Exception as emsg:
                self.echo("Saving failed: %s"%emsg)
        
        imLeft = self.plotLeft.getImage()
        imRight = self.plotRight.getImage()
        oldData = imLeft.getData()
        imRight.setData(oldData)
        imLeft.setData(img)
        self.plotLeft.resetZoom()

    def calibrate(self, motorNum):
        motorName = self.control.Input["Mot%i"%motorNum].text()#.toAscii()
        motorStep = float(self.control.Input["Step%i"%motorNum].text())#.toFloat()[0]
        imLeft = self.plotLeft.get_roi_data()
        imRight = self.plotRight.get_roi_data()
        
        sa = sift.LinearAlign(imLeft, devicetype='GPU')
        res = sa.align(imRight, shift_only=True, return_all=True,
                       double_check=False, relative=False, orsa=False)
        
        if res is None or res["matrix"] is None or res["offset"] is None:
            self.echo("Warning: No matching keypoints found.")
            return
        
        self.plot_matchpoints(res)
        
        offset = -res["offset"][::-1]
        #print(offset)
        output = "Offset estimated for %s movement of %f: (%.2f, %.2f) px" \
             %(motorName, motorStep, offset[0], offset[1])
        self.echo(output)
        dv_vs_dm = old_div(offset, motorStep)
        
        
        self.control.Input["CalRes_%i"%motorNum].setText("%.2f, %.2f"%tuple(dv_vs_dm))
        self.calibration[motorNum] = dv_vs_dm

    def get_center_of_rotation(self):
        imLeft = self.plotLeft.get_roi_data().astype(float)
        imRight = self.plotRight.get_roi_data().astype(float)
        
        if not imLeft.size or not imRight.size:
            self.echo("Error: ROI outside image data.")
            return
        
        roi = self.plotLeft.roi
        dx, dy = np.array(roi)[:,0] if roi is not None else (0,0)
        
        #print(imLeft.shape, imRight.shape, roi)
        
        #sigma = float(self.control.Input["sigma"].text().toFloat()[0])
        sigma = 1.6 # default value
        t0 = time.time()
        try:
            sa = sift.LinearAlign(imLeft, devicetype='GPU',init_sigma=sigma)
            res = sa.align(imRight, shift_only=False, return_all=True, 
                           double_check=False, relative=False, orsa=False)
        except Exception as emsg:
            self.echo("Error during alignment: %s"%emsg)
            return
        self.echo("Calculation time: %.2f ms"%((time.time() - t0)*1000))
        self.resultsCOR = dict(align=res)
        
        if res is None or res["matrix"] is None or res["offset"] is None:
            self.echo("Warning: No matching keypoints found.")
            return
        
        self.plot_matchpoints(res)
        numpoints = len(res["matching"])
        
        
        if numpoints<18:
            self.echo("Too few matching keypoints found (%i)."%numpoints)
            return
        
        self.echo("Matching keypoints found: %i"%numpoints)
        
        matrix, offset = res["matrix"][::-1,::-1], res["offset"][::-1]
        #offset[0] += dx
        #offset[1] += dy
        
        U, S, V = linalg.svd(matrix)
        R = U.dot(V) # Rotation part
        self.resultsCOR.update(dict(U=U, S=S, V=V, R=R))
        relrot = abs(R[0,0] - 1)
        if relrot < 1e-3:
            self.echo("Estimation of rotation failed. Too small? (%.3g)"%relrot)
            return
        angle = np.degrees(np.arctan2(R[0,1], R[1,1]))
        self.echo("Rotation of %.2f deg found."%angle)
        
        
        cor = linalg.solve(matrix - np.eye(2), -offset).squeeze()
        cor[0] += dx
        cor[1] += dy
        self.resultsCOR["cor"] = cor
        self.echo("Center of rotation estimated at (%.2f, %.2f) px."%tuple(cor))
        
        plotcfg = dict(symbol="o", legend="cor", 
                        color=(1.,.3,.3,1.), text="COR")
        self.plotLeft.addMarker(cor[0], cor[1], **plotcfg)
        self.plotRight.addMarker(cor[0], cor[1], **plotcfg)
    
    
    def calc_sharpness(self):
        imLeft = self.plotLeft.get_roi_data().astype(float)
        sharpness = image.contrast(imLeft)
        self.echo("Computed sharpness of the left image ROI: %f"%sharpness)
        
    def plot_matchpoints(self, res):
        roi = self.plotLeft.roi
        dx, dy = np.array(roi)[:,0] if roi is not None else (0,0)
        
        xk1 = res["matching"].x[:,0] + dx
        xk2 = res["matching"].x[:,1] + dx
        yk1 = res["matching"].y[:,0] + dy
        yk2 = res["matching"].y[:,1] + dy
        
        self.plotLeft.update_keypoints((xk1, yk1))
        self.plotRight.update_keypoints((xk2, yk2))
    
    
    def poi_to_cor(self):
        poi = self.plotLeft.poi 
        cor = self.resultsCOR.get("cor", None)
        if poi is None:
            self.echo("Use crosshair to select point of interest first.")
            return
        if cor is None:
            self.echo("Error: No center of rotation found.")
            return
            
        diff = cor - poi
        self.echo("Distance: (%.2f, %.2f) px"%tuple(diff))
        
        calibration = []
        for i in (1,2):
            if not i in self.calibration:
                self.echo("Motor %i not calibrated"%i)
                return
            calibration.append(self.calibration[i])
        
        
        matrix = np.linalg.inv(np.array(calibration).T)
        
        dm1, dm2 = matrix.dot(diff)
        m1, m2 = [self.control.Input["Mot%i"%i].text() for i in (1,2)]
        
        self.echo("Move to POI:")
        self.echo("  umvr %s %s"%(m1, dm1))
        self.echo("  umvr %s %s"%(m2, dm2))

    def echo(self, output):
        self.control.Input["output"].append(output)
    
    def close_application(self):
        choice = Q.QMessageBox.question(self, 'Quit',
                                "Do you really want to quit?",
                                Q.QMessageBox.Yes | Q.QMessageBox.No)
        if choice == Q.QMessageBox.Yes:
            sys.exit()
        else:
            pass


def run():
    app = Q.QApplication(sys.argv)
    #app.setStyle("CleanLooks")
    GUI = Window()
    sys.exit(app.exec_())
    #app.exec_()

if __name__=="__main__":
    
    run()
