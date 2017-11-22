import numpy as np


class AreaDetector(object):
    """
        The Base class.
        This is temporary until proper nexus classes are implemented
    """
    def __init__(self, directions, pixsize, pixnum, mask=None):
        if np.isscalar(pixsize):
            pixsize = (pixsize, pixsize)
        if np.isscalar(pixnum):
            pixnum = (pixnum, pixnum)
        self.directions = directions
        self.pixsize = pixsize
        self.pixnum  = pixnum
        self.mask = mask
    
    @staticmethod
    def correct_image(image):
        pass # to be implemented for each detector


class MaxiPix(AreaDetector):
    def __init__(self, mask=None):
        super(MaxiPix, self).__init__(directions=("z-", "y+"),
                                      pixsize=55e-6,
                                      pixnum=516,
                                      mask=mask
                                     )
    @staticmethod
    def correct_image(image):
        """
            Mind the gap.
        """
        image *= 9
        image[255:258] = image[255]/3
        image[258:261] = image[260]/3
        image[:,255:258] = (image[:,255]/3)[:,None]
        image[:,258:261] = (image[:,260]/3)[:,None]


