import collections


class test(collections.MutableMapping):
    def __init__(self, testdict):
       self.testdict = testdict
       self.store = dict()
       self.update(self.testdict)

    def __getitem__(self, key):
        return self.store[key]*2
    
    def __setitem__(self, key, value):
        #print('WARNING: you are changing a linked lookuptable')
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


    
class lookupdict_rho_theta(collections.MutableMapping):
    def __init__(self,stage):
       self.stage = stage
       self.store = dict()
       self.update(self.stage.lookup['rho'])
       if 'z' not in list(self.store.keys()):
           self.update({'z':np.zeros(shape = self.store['x'].shape)})

    def __getitem__(self, key):
        '''
        here we can recalulate all values for the lookup as a function of what ever stage gibes us
        '''
        theta_rad = stage.wm('theta')/180.0*np.pi
        if key == 'x':
            return self.store[key]*np.cos(theta_rad)
        if key == 'z':
            return self.store[key]*np.sin(theta_rad)
        
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
       
class TransformedDict(collections.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key
