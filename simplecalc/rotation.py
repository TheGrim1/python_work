import numpy as np


def poni_rot_to_xrayutilities_tilt(rot1,rot2):
    '''
    returns tiltazimuth, tilt in degree
    accepts rot1, rot2 from the pyFAI ponifiles in radians
    '''
    tiltazimuth = np.arctan(rot1/rot2)*180/np.pi
    tilt = np.sqrt(rot1**2+rot2**2)*180/np.pi
    return tiltazimuth, tilt
    
