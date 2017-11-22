import collections
import numpy as np

'''
This collection comprises some modified lookuptables used by CamView_stages
The lookuptables are supposed to look like a pyhton dict object. They will be addressed in the stage instance under self.lookup[<key>] -> lookupdict
where key is the motor that is moving using a lookuptable. The keys in loolkuptables must contain the moving motor and all motors that will be used to correct this movement.
The values behind these keys must be np.arrray-like and have the same shape for all motors so each motor has a vlaue for all reference positions of the moving motor. Specifically the excecution is:
<< from CamView_tools: >>
    def _correct_with_lookup(self, function, start_pos, end_pos):
        print 'correcting movement of %s with motors:' % function
        if function in self.lookup.keys():
            for mot in self.lookup[function].keys():
                if mot != function:
                    start_correction = np.interp(start_pos, self.lookup[function][function], self.lookup[function][mot])
                    end_correction   = np.interp(end_pos, self.lookup[function][function], self.lookup[function][mot])
                    correction = end_correction-start_correction
                    self.mvr(mot, correction)
        else:
            print 'no lookuptable found for ' , function
'''

class lookupdict_rho_yztheta(collections.MutableMapping):
    '''
    corrects the rotation around <rho> when the axis of rho is tilted by theta in the y,z-plane
    '''
    def __init__(self,stage):
        self.stage = stage
        self.store = dict()
        self.update(self.stage.lookup['rho'])
        if 'z' not in self.store.keys():
            self.update({'z':np.zeros(shape = self.store['x'].shape)})
        
           

    def __getitem__(self, key):
        '''
        here we can recalulate all values for the lookup as a function of what ever self.stage gives us as a theta value before returning the lookup values. This saves recording a 2d lookup table.
        '''
        if self.stage.stagegeometry['COR_motors']['theta']['invert']:
            theta_rad = - self.stage.wm('theta')/180.0*np.pi
        else:
            theta_rad = self.stage.wm('theta')/180.0*np.pi
        
        if key == 'y':
            return self.store['y']*np.cos(theta_rad) - self.store['z']*np.sin(theta_rad)
        elif key == 'z':
            return self.store['y']*np.sin(theta_rad) + self.store['z']*np.cos(theta_rad)
        else:
            return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key


class lookupdict_phi_yzkappa(collections.MutableMapping):
    '''
    corrects the rotation around <rho> when the axis of rho is tilted by kappa in the y,z-plane
    '''
    def __init__(self,stage):
        self.stage = stage
        self.store = dict()
        self.update(self.stage.lookup['phi'])
        if 'z' not in self.store.keys():
            self.update({'z':np.zeros(shape = self.store['x'].shape)})
        
           

    def __getitem__(self, key):
        '''
        here we can recalulate all values for the lookup as a function of what ever self.stage gives us as a kappa value before returning the lookup values. This saves recording a 2d lookup table.
        '''
        if self.stage.stagegeometry['COR_motors']['kappa']['invert']:
            kappa_rad = - self.stage.wm('kappa')/180.0*np.pi
        else:
            kappa_rad = self.stage.wm('kappa')/180.0*np.pi
        
        if key == 'y':
            return self.store['y']*np.cos(kappa_rad) - self.store['z']*np.sin(kappa_rad)
        elif key == 'z':
            return self.store['y']*np.sin(kappa_rad) + self.store['z']*np.cos(kappa_rad)
        else:
            return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key



class lookupdict_phi_xzkappa(collections.MutableMapping):
    '''
    corrects the rotation around <rho> when the axis of rho is tilted by kappa in the y,z-plane
    '''
    def __init__(self,stage):
        self.stage = stage
        self.store = dict()
        self.update(self.stage.lookup['phi'])
        if 'z' not in self.store.keys():
            self.update({'z':np.zeros(shape = self.store['x'].shape)})
        elif not self.store['z'].shape == self.store['x'].shape:
            self.update({'z':np.zeros(shape = self.store['x'].shape)})

    def __getitem__(self, key):
        '''
        here we can recalulate all values for the lookup as a function of what ever self.stage gives us as a kappa value before returning the lookup values. This saves recording a 2d lookup table.
        '''
        if self.stage.stagegeometry['COR_motors']['kappa']['invert']:
            kappa_rad = - self.stage.wm('kappa')/180.0*np.pi
        else:
            kappa_rad = self.stage.wm('kappa')/180.0*np.pi
        
        if key == 'x':
            return self.store['x']*np.cos(kappa_rad) - self.store['z']*np.sin(kappa_rad)
        elif key == 'z':
            return self.store['x']*np.sin(kappa_rad) + self.store['z']*np.cos(kappa_rad)
        else:
            return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key

