# collect all LookupDict classes here. These act like dictionaries but can have 'dynamic' interdependencies between motors.
# eg. LookupDict_Phi_XZKappa where the lookup for phi (x-y-z) is rotated by kappa (x-z) components are mixed accordingly.
# To make your custom class: overwrite getitem with the desired function
# requires passing a motors dict with {mot_name:{invert:[bool],is_rotation:[bool]}



from __future__ import print_function
import collections
import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class LookupDict(collections.MutableMapping):
    def __init__(self, motors_dc):
        self.currpos = dict()
        self.motors = motors_dc
        self.store = dict()
        
    def __getitem__(self, key):
        '''
        here a calcutation of interdependent lookuptables (eg. phi - kappa) can be done
        '''
        
        return self.store[key]

    def __setitem__(self, key, value):
        '''
        TODO: use setitem inteligently
        here a calcutation of interdependent lookuptables (eg. phi - kappa) could be done
        '''
        self.store[key] = value

    
    def mockup_currpos(self, currpos):
        '''
        update the internal position dictionary with curpos : {mot1_name:mot1_pos .. etc}
        '''
        # print("mockup:", currpos)
        self.currpos.update(currpos)

    def wm(self, function):
        '''
        retuns internaly stored position
        '''
        return self.currpos[function]


    ## The rest is required for the class to act like a dict
        
    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key


class LookupDict_Phi_XZKappa(LookupDict):
    '''
    child of LookupDict that corrects the rotation around <phi> when the axis of phi is tilted by kappa in the x,z-plane
    '''
    def __init__(self, motors, lookup):
        self.currpos = dict()
        self.motors = motors
        self.store = dict()
        self.update(lookup['phi'])
        if 'z' not in list(self.store.keys()):
            self.update({'z':np.zeros(shape = self.store['x'].shape)})

    def __getitem__(self, key):
        if self.motors['kappa']['invert']:
            kappa_rad = self.wm('kappa')/180.0*np.pi
        else:
            kappa_rad = - self.wm('kappa')/180.0*np.pi
            
        if key == 'x':
            print("key y; kappa_rad =", kappa_rad)
            return self.store['x']*np.cos(kappa_rad) - self.store['z']*np.sin(kappa_rad)
        elif key == 'z':
            print("key z; kappa_rad =", kappa_rad)
            return self.store['x']*np.sin(kappa_rad) + self.store['z']*np.cos(kappa_rad)
        else:
            return self.store[key]



class LookupDict_Phi_hexXZKappa(LookupDict):
    '''
    child of LookupDict that corrects the rotation around <phi> when the axis of phi is tilted by kappa in the x,z-plane
    '''
    def __init__(self, motors, lookup):
        self.currpos = dict()
        self.motors = motors
        self.store = dict()
        self.update(lookup['phi'])
        if 'hex_z' not in list(self.store.keys()):
            self.update({'hex_z':np.zeros(shape = self.store['hex_x'].shape)})

    def __getitem__(self, key):
        if self.motors['kappa']['invert']:
            kappa_rad = self.wm('kappa')/180.0*np.pi
        else:
            kappa_rad = - self.wm('kappa')/180.0*np.pi
            
        if key == 'hex_x':
            print("key {}; kappa_rad = {}".format(key,kappa_rad))
            return self.store['hex_x']*np.cos(kappa_rad) - self.store['hex_z']*np.sin(kappa_rad)
        elif key == 'hex_z':
            print("key {}; kappa_rad = {}".format(key,kappa_rad))
            return self.store['hex_x']*np.sin(kappa_rad) + self.store['hex_z']*np.cos(kappa_rad)
        else:
            print("key {}; no kappa correction".format(key))
            return self.store[key]



        
    
class Lut_TOMO_Phi_PhiKappa2D(LookupDict):
    '''
    child of LookupDict that corrects the rotation around <phi> when the axis of phi is tilted by kappa in the x,z-plane
    '''
    def __init__(self, motors, lookup):
        self.currpos = dict()
        self.motors = motors
        self.store = dict()
        self.update(lookup['phi'])
        if 'z' not in list(self.store.keys()):
            self.update({'hex_z':np.zeros(shape = self.store['x'].shape)})

    def __getitem__(self, key):
        kappa_val = self.wm('kappa') % 360
        kappa_list = [x for x in self.store.keys() if type(x)==np.int32]

        kappa_nearest = find_nearest(np.asarray(kappa_list),kappa_val)
        info = "key {}; kappa_val = {}; nearest lookup: {}".format(key,kappa_val,kappa_nearest)
            
        return self.store[kappa_nearest][key]

    def add_phi_for_kappa(kappa_val,phi_lookupdict):
        
        kappa_val = kappa_val%360
        # doubles the lookuptable, but handles cases where kappa < 0 or kappa > 360:
        self.update({np.int(kappa_val):phi_lookup_dict})
        self.update({np.int(kappa_val+360):phi_lookup_dict})


