import numpy as np


def save_data(fname, data, header = None, delimiter = ' ', quotechar = '#'):
    '''
    counterpart to open_data,
header can be str or list type
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

def open_data(filename, delimiter = ' ', quotecharlist= ['#'],verbose = False):
    '''reads <filename> as a <delimiter> seperated datafile and returns the data as np.array \n ignores lines staring with something in quotecharlist '''
    
    data = []
    f = open(filename, 'r')
    reader = f.readlines()
    
    if verbose:
        print('read lines:')

    header = []
    for i, l in enumerate(reader):
        try:            
            if l.lstrip()[0] in quotecharlist:
                header.append((l[1:].rstrip().split(delimiter)))
            else:
                # print 'found line '
                # print l
                # print l.lstrip().split(delimiter)
                # print 'parsed it as :'
                # print [float(x) for x in (l.rstrip().split(delimiter)) if len(x)> 0]
                data.append([float(x) for x in (l.rstrip().split(delimiter)) if len(x)> 0])
        except IndexError:
            if verbose:
                print('discarding:')
                print(l)

    if verbose:
        print(data)

    if len(header) ==1:
        header = header[0]
            
    data = np.asarray(data)
    return data, header
