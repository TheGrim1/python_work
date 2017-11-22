from builtins import object
import collections
import xrayutilities as xu


class EmptyGeometry(object):
    """
        Abstract container for diffractometer angles
    """
    sample_rot = collections.OrderedDict()
    detector_rot = collections.OrderedDict()
    offsets = collections.defaultdict(float)
    
    # defines whether these motors are used. otherwise set to zero.
    usemotors = set()
    
    inc_beam = [1,0,0]
    
    def __init__(self, **kwargs):
        """
            Initialize diffractometer geomtry.
            
            Inputs: all motor names from self.sample_rot and self.detector_rot
                True  -- use motor
                False -- discard
        """
        usemotors = self.usemotors
        for motor in kwargs:
            usemotors.add(motor) if kwargs[motor] else usemotors.discard(motor)
        
    def getQconversion(self, inc_beam = None):
        if inc_beam is None:
            inc_beam = self.inc_beam
        qc = xu.experiment.QConversion(list(self.sample_rot.values()),
                                       list(self.detector_rot.values()),
                                       inc_beam
                                      )
        return qc
    
    def set_offsets(self, **kwargs):
        """
            Set offset for each motor to be subtracted from its position.
            Motors identified by keyword arguments.
        """
        for kw in kwargs:
            if kw in self.sample_rot or kw in self.detector_rot:
                self.offsets[kw] = float(kwargs[kw])



class ID01psic(EmptyGeometry):
    def __init__(self, **kwargs):
        ### geometry of ID01 diffractometer
        ### x downstream; z upwards; y to the "outside" (righthanded)
        ### the order matters!
        self.sample_rot['mu'] = 'z-' # check mu is not 0
        self.sample_rot['eta'] = 'y-'
        self.sample_rot['phi'] = 'z-'
        self.sample_rot['rhy'] = 'x-' # can be useful to correct sample tilt?!
        self.sample_rot['rhx'] = 'y+' # can be useful to correct sample tilt?!
    
        self.detector_rot['nu'] = 'z-'
        self.detector_rot['delta'] = 'y-'
        
        self.inc_beam = [1,0,0]
        
        # defines whether these motors are used. otherwise set to zero
        #   typical defaults, can be overridden during __init__:
        self.usemotors = set(('eta', 'phi', 'nu', 'delta'))
        super(ID01psic, self).__init__(**kwargs)
    

