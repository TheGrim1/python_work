from __future__ import print_function
import os,time,sys
import struct
import numpy
import numpy as np


os.environ['QUB_SUBPATH'] = 'qt4'
from PyQt4 import QtGui,QtCore
from PyQt4.QtCore import Qt

from PyTango import DeviceProxy   # better to use PyTango.gevent ?

# from bliss.data.routines.pixmaptools import qt4 as pixmaptools
import pixmaptools.qt4 as pixmaptools


#print "set video_live TRUE"
#device.video_live=True

lutMode = pixmaptools.LUT.Scaling.YUV422PACKED

class error(Exception): pass

class CameraProxy(object):

    DEFAULT_DEVNAME = "id13/limaccds/eh3-vlm1"


    def __init__(self, devname='<default>'):
        if devname == '<default>':
            devname = self.DEFAULT_DEVNAME

        self.devname = devname
        self.device = DeviceProxy(self.devname)
        self.j = 0

    def show_devinfo(self):
        device = self.device
        print("tango device=",     device.name())
        print("Exposure Time=",    device.acq_expo_time)
        print("camera_model=",     device.camera_model)
        print("camera_pixelsize=", device.camera_pixelsize)
        print("camera_type=",      device.camera_type)
        print("image_height=",     device.image_height)
        print("image_width=",      device.image_width)
        
        print("last_image_acquired =", device.last_image_acquired)
        print("video_mode =", device.video_mode)
        print("video_live =", device.video_live)

    def set_live(self):
        device = self.device
        device.video_live=True

    def acquire_qimage(self):
        device = self.device
        image_data = device.video_last_image
        if not self.j % 50:
            print("cycle:", self.j, "last_image_acquired =", device.video_last_image_counter)
        if image_data[0]=="VIDEO_IMAGE":
            header_fmt = ">IHHqiiHHHH"
            header_size= struct.calcsize(header_fmt)
            _, ver, img_mode, frame_number, width, height, _, _, _, _ = struct.unpack(header_fmt, image_data[1][:header_size])
            #print "ver=%r, img_mode=%r, frame_number=%r, width=%d, height=%d" % (ver, img_mode, frame_number, width, height)
            self.shape = (height, width)
            raw_buffer = numpy.fromstring(image_data[1][header_size:], numpy.uint16)
        else:
            print("ERROR : No header found")
            raise error("image acquisition failed")


        scaling = pixmaptools.LUT.Scaling()
        scaling.autoscale_min_max(raw_buffer, width, height, lutMode)
        # scaling.set_custom_mapping(12 , 50)
    
        returnFlag,qimage =  pixmaptools.LUT.raw_video_2_image(raw_buffer, width, height, lutMode, scaling)

        self.j += 1
        return (returnFlag,qimage, device.video_last_image_counter)

    def convert_to_greyscale_int18(self, qimage):
        qimg = qimage.convertToFormat(QtGui.QImage.Format_RGB32)
        qiarr = qimage2ndarray.recarray_view(qimg)
        redrr   = qiarr['r']
        greenrr = qiarr['g']
        bluerr  = qiarr['b']
        sumrr   = redrr.astype(np.int16)
        sumrr   += greenrr
        sumrr   += bluerr
        sumrr   = sumrr.reshape(self.shape)
        return qimg, qiarr, redrr, greenrr, bluerr, sumrr

    def acquire_greyscale_int18(self):
        err_flg, qimage, last_img_num = self.acquire_qimage()
        qimg, qiarr, redrr, greenrr, bluerr, sumrr = self.convert_to_greyscale_int18(qimage)
        return sumrr, last_img_num

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
        errflg, qimage, last_img_num = self.cp.acquire_qimage()

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



def main():
    args = sys.argv[1:]
    devn = args[0]
    if not int(devn) in (1,2):
        print("camera selection can be only 1 or 2")
        sys.exit(1)
    devname = "id13/limaccds/eh3-vlm%1d" % int(devn)
    cp = CameraProxy(devname=devname)
    cp.set_live()
    v = Viewer(cp=cp, num=int(devn))
    v.gui_run()

if __name__ == '__main__':
    main()
