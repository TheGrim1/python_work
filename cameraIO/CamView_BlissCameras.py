from __future__ import print_function
from __future__ import absolute_import

sys.path.append('/data/id13/inhouse2/AJ/skript')
from simplecalc.slicing import troi_to_slice
from fileIO.images.image_tools import uint8array_to_qimage

class BlissCameras(object):
    def __init__(self, cameralist):
        self.cameras = cameralist
        self.image_counters = [0 for camera in self.cameras]

    def __getitem__(self,i):
        return self.grab_image(cam_no=i, troi=None)

    def grab_image(self, cam_no, troi=None):
        camera = self.cameras[cam_no]
        ct_func = camera[0]
        counter = camera[1]
        exp_time = camera[2]
        scan = ct_func(exp_time, counter) 
        arr = np.asarray(scan.get_data()["{}:image".format(counter.name)])

        if type(troi)==type(None):
            return arr
        else:
            return arr[troi_to_slice(troi)]

        
    def grab_qimage(self, cam_no, troi=None):
        array = self.grab_image(cam_no,troi)

        #from https://www.swharden.com/wp/2013-06-03-realtime-image-pixelmap-from-numpy-array-data-in-qt/
        array = numpy.require(array, numpy.uint8, 'C')
        qimage = uint8array_to_qimage(im)
        return qimage
