from __future__ import print_function
import os,time,sys
import struct
import numpy as np


os.environ['QUB_SUBPATH'] = 'qt4'
from PyQt4 import QtGui,QtCore
from PyQt4.QtCore import Qt

from PyTango import DeviceProxy   # better to use PyTango.gevent ?


class error(Exception): pass

class Viewer(object):

    def __init__(self, cp=None, num=1):
        self.num = num
        self.cp = cp
        self.app = QtGui.QApplication(sys.argv)  # main application
        self.label = label = QtGui.QLabel()
        self.timer = QtCore.QTimer(label)
        self.painter = QtGui.QPainter()
        self.marker = Marker((748,576),10,10, self.painter)
        self.pos = (345,456)
        self.i = 0

    def gui_run(self):
        app = self.app
        timer = self.timer
        label = self.label
        label.resize(748, 576)

        QtCore.QObject.connect(timer,QtCore.SIGNAL('timeout()'), self.refresh)
        timer.start(50);
        label.show()
        app.exec_()
        timer.stop()

    def refresh(self):
        painter = self.painter
        errflg, qimage, last_img_num = self.cp.grab_qimage()

        if self.i % 15:
            try:
                f = file('marker.txt')
                ll = f.readlines()
                l = ll[self.num]
                f.close()
                pos = p0,p1 = list(map(int, l.split()))
            except:
                pos = 20,20
                
            self.marker.set_pos(pos)
        
        self.i += 1

        painter.begin(qimage)
        painter.setPen(QtGui.QPen(Qt.red)) 
        self.marker.paint()
        #rect = QtCore.QRect(50, 50, 400, 400)
        #painter.drawRect(rect)
        #painter.drawRect()
        #painter.drawLine(200,230,240,270)
        painter.end()
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimage))


class Marker(object):

    def __init__(self, frame_shape, hgap, llen, painter):
        self.frame_shape = frame_shape
        self.hgap = hgap
        self.llen = llen
        self.painter = painter
        self.set_pos((100,100))

    def set_pos(self, pos):
        self.pos = pos
        self.make_coords()

    def make_coords(self):
        g = self.hgap
        l = self.llen
        p0,p1 = self.pos
        ll = []
        ll.append((p0-g-l,p1,p0-g,p1))
        ll.append((p0+g,p1,p0+g+l-1,p1))
        ll.append((p0,p1-g-l,p0,p1-g-1))
        ll.append((p0,p1+g-1,p0,p1+g+l-1))
        self.ll = ll

    def paint(self):
        painter = self.painter
        ll = self.ll
        for x in ll:
            painter.drawLine(*x)


def launch_live_viewer(devname):
    '''
    devname = "id13/limaccds/eh2-vlm%1d" % int(devnumber)
    or
    devname = "USB%1d" % int(devnumber)
    '''
    if devname[:3].upper() == 'USB':
        if 'USBCameras' not in dir():
            print('inporting USBCameras ')
            import CamView_USBCameras.USBCameras as USBCameras
        devnum= int(devname[3])
        cp = USBCameras()
        test = cp.grab_image(devnum)
        print('test image shape = ', test.shape)
        
    else:
        if 'ETHCameras' not in dir():
            print('inporting ETHCameras ')
            import CamView_ETHCameras.ETHCameras as ETHCameras
        cp = ETHCameras([devname])
        # get test image to init:
        devnum = 1
        test = cp.grab_image(devnum)
        print('test image shape = ', test.shape)

        
    v = Viewer(cp=cp, num=int(devnum))
    v.gui_run()


def main():
    args = sys.argv[1:]
    devn = args[0]
    if not int(devn) in (1,2):
        print("camera selection can be only 1 or 2")
        sys.exit(1)
    devname = "id13/limaccds/eh2-vlm%1d" % int(devn)
    cp = CameraProxy(devname=devname)
    cp.set_live()
    v = Viewer(cp=cp, num=int(devn))
    v.gui_run()

if __name__ == '__main__':
    main()
