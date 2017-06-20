import os,sys
import numpy as np

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
from fileIO.datafiles.open_data import open_data


def save_data(fname, data, header = None, delimiter = ' ', quotechar = '#'):
    '''
    counterpart to open_data,
header can be str od list type
    '''

    f = open(fname, 'w')

    if type(header) is list:
        f.write('#' + delimiter.join(header) + '\n')
    elif type(header) is str:
        f.write('#' + header + '\n')
        
        
    for dataline in data:
        f.write(delimiter.join([str(x) for x in dataline]) + '\n')
    
    f.flush()
    f.close()

def test():
    testdata = np.random.randn(5,3)

    fname = './readtest.txt'
    
    testheader = ['more bla']
    
    save_data(fname, testdata, testheader, delimiter = ',')
    
    readdata = open_data(fname, delimiter = ',')

    fname = './savetest.txt' 
    save_data(fname, testdata, testheader, delimiter = '\t')
    
    print 'if readtest.txt = savetest.txt - SUCCESS'



if __name__ == '__main__':
    test()
