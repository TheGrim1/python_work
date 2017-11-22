from __future__ import print_function
from builtins import object
import numpy as np
from silx.io.spech5 import SpecH5
import timeit


# only needed for testing:
import matplotlib.pyplot as plt

class spec_mesh(object):
    '''
    handles spec mesh scans for saving and opening etc.
    '''
    def __init__(self,
                 fname = "/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/DATA/AJ2c_after/AJ2c_after.dat",
                 scanno = 318,
                 counter = 'ball01'):
        
        self.info = {}
        self.data = np.zeros(shape = (0,0))
        self.info.update({'fname'  :fname})
        self.info.update({'counter': counter})
        self.info.update({'scanno' : scanno})

    def load(self):
        sfh5        = SpecH5(self.info['fname'])
        grouptpl    = '%s.1/'
        speccommand = sfh5[grouptpl % self.info['scanno']]['title']
        if not (speccommand.split()[1]) == 'mesh':
            print( '\nThis scan is not a 2D mesh, cant initiate data from scan np %s !\n' % self.info['scanno'])
            raise ValueError

        scanshape   = (int(speccommand.split()[9]) + 1, int(speccommand.split()[5]) + 1)
        motornames  = (speccommand.split()[2],speccommand.split()[6])
        realshape   = (abs(float(speccommand.split()[3]) - float(speccommand.split()[4])),abs(float(speccommand.split()[7]) - float(speccommand.split()[8])))
        exptime     = float(speccommand.split()[10])
        
        data        = np.zeros(shape = (scanshape[0],scanshape[1]))
        flat        = sfh5[grouptpl % self.info['scanno']]['measurement'][self.info['counter']]
        data        = np.reshape(flat, scanshape)

        try:
            Theta = sfh5[grouptpl % self.info['scanno']]['instrument']['positioners']['Theta']
        except KeyError:
            Theta = 'KeyError'

        self.data                      = data
        self.info.update({'shape'      : scanshape})
        self.info.update({'motornames' : motornames})
        self.info.update({'realshape'  : realshape})
        self.info.update({'exptime'    : exptime})
        self.info.update({'Theta'      : Theta})

def open_scan(fname = "/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/DATA/AJ2c_after/AJ2c_after.dat",
              scanlist  = [318],
              counter = 'ball01'):
    '''opens counter of all scans in scanlist. Reshapes the scans according to the title. Returns just data'''

#preps:
    sfh5        = SpecH5(fname)
    grouptpl    = '%s.1/'
    speccommand = sfh5[grouptpl %scanlist[0]]['title']
    scanshape   = (int(speccommand.split()[9]) + 1, int(speccommand.split()[5]) + 1)
    data        = np.zeros(shape = (scanshape[0],scanshape[1], len(scanlist)))
    
    i = 0
    for scan in scanlist:
        print('reading scan no %s' %scan)

        # #    timing:
        # start_time = timeit.default_timer()
        # print 'took %s' % (timeit.default_timer() - start_time)
        
        flat        = sfh5[grouptpl % scan]['measurement'][counter]
        data[:,:,i] = np.reshape(flat, scanshape)
        i          +=1
    return data

def open_dscans(fname = "/data/id13/inhouse6/THEDATA_I6_1/d_2016-11-17_inh_ihsc1404/DATA/AJ2c_after/AJ2c_after.dat",
                scanlist  = [333], # 333 +16 und nnp2 bei 313 +16
                counter = 'Detector',
                sorting_motor=None):
    '''opens counter of all dscans in scanlist. Stacks scans. Returns just data. They must be the same length for this to make sense'''

#preps:
    sfh5        = SpecH5(fname)
    grouptpl    = '%s.1/'
    speccommand = sfh5[grouptpl %scanlist[0]]['title']
    print(speccommand.split())
    scanlen     = int(speccommand.split()[5]) +1
    scan_positions = (float(speccommand.split()[3]) - float(speccommand.split()[4])) * np.arange(float(scanlen))/float(scanlen) + float(speccommand.split()[3])
    data        = np.zeros(shape = (len(scanlist),scanlen))
    at_positions = []
    
    i = 0
    for i,scan in enumerate(scanlist):
        print('reading scan no %s' %scan)

        # #    timing:
        # start_time = timeit.default_timer()
        # print 'took %s' % (timeit.default_timer() - start_time)

        if type(sorting_motor) == str:
            at_positions.append(float(sfh5[grouptpl %scan]['instrument']['positioners'][sorting_motor]))

        # print (np.asarray(sfh5[grouptpl % scan]['measurement']['Detector']))
        data[i]     = np.asarray(sfh5[grouptpl % scan]['measurement'][counter])
        
       
    return data , np.asarray(at_positions), scan_positions 



if __name__ == "__main__":
    'test'
    
    data = open_scan()
        
    plt.imshow(data[0])
    plt.show()
