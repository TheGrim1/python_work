'''
many thanks to Carsten Richter for inspiration (copying id01's CamView)
'''
from __future__ import print_function
from __future__ import absolute_import

import sys, os
import time
import silx
from silx.gui import plot
from silx.gui.plot import PlotActions

from PyQt4 import QtGui as Q
from PyQt4 import QtCore
print("Using PyQt %s"%QtCore.QT_VERSION_STR)

import numpy as np

from .baslertools_aj import launch_live_viewer

class GoLiveAction(PlotActions.PlotAction):
    def __init__(self, plot, parent=None):
        super(GoLiveAction, self).__init__(
            plot,
            icon='camera',
            text='toggle live view',
            tooltip='Toggles live view on/off for this view',
            triggered=self.trigger,
            checkable=True,
            parent=parent)

    def trigger(self):
        if self.isChecked():
            launch_live_viewer("id13/limaccds/eh2-vlm1")
            print('was checked')
            
        else:
            print('was unchecked')
        

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
        
        golive = GoLiveAction(self)
        self.toolBar().addAction(golive)
        self._golive = golive
        
        
class Window(Q.QMainWindow):
    _ignoreEvent = False
    _eventSource = None

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(200, 100, 1000, 750)
        self.setWindowTitle("Micro centereing stage")
        
        extractAction = Q.QAction("&Quit", self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip('Leave The App')
        extractAction.triggered.connect(self.close_application)
        
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

        
    def home(self):
        cw = Q.QWidget(self)
        self.grid = g = Q.QGridLayout(cw)
        self.setCentralWidget(cw)

        data = np.random.random((512,512))
        self.plotLeft  = pleft  = CamPlot(data, "live view 01", cw)
        g.addWidget(pleft,  0,0)

        self.show()

        
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
