"""
    This is the file where detectors are defined.
    Todo: Andor and Eiger
"""
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
        image *= 9   # TODO: warum? the pixel at the edge has 3 times more counts, should be distributed to the other pixels.
        image[255:258] = image[255]/3
        image[258:261] = image[260]/3
        image[:,255:258] = (image[:,255]/3)[:,None]
        image[:,258:261] = (image[:,260]/3)[:,None]

    @staticmethod
    def ff_correct_image(image):
        """
            perhaps a flatfield here
        """
        pass


class Eiger2M(AreaDetector):
    def __init__(self, mask=None):
        super(Eiger2M, self).__init__(directions=("z-", "y+"),
                                      pixsize=75e-6,
                                      pixnum=(2164,1030),
                                      mask=mask
                                     )
    @staticmethod
    def ff_correct_image(image):
        """
            perhaps a flatfield here
        """
        pass

    @staticmethod
    def mask_image(image):
        """
            Mind the BIG gaps and the bad columns
        """
        pass

