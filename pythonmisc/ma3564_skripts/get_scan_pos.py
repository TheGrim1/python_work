from silx.io.spech5 import SpecH5
import numpy as np
import matplotlib.pyplot as plt
import sys


def main(fname):

    scanno_list = [(x*3)+199 for x in range(101)]

    sfh5 = SpecH5(self.info['fname'])

    grouptpl    = '%s.1/'
    
    for i, scanno in scanno_list:
        title = sfh5[grouptpl.format(scanno)].title
        print('scan {} cmd :{}'.format(scanno, title))
    
    
if __name__ == '__main__':
    fname = sys.argv[1]
    main(fname)
