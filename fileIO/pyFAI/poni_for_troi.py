import numpy
import sys, os
import time
        
def initiate_ponidict():
    parameterlist = ['Detector','PixelSize1','PixelSize2','Distance','Poni1','Poni2','Rot1','Rot2','Rot3','Wavelength']
    ponidict = {}
    for param in parameterlist:
        ponidict.update({param:'empty'})
    return ponidict

def ponidict_full(ponidict):
    full = True
    for item in list(ponidict.items()):
        if item[1] == 'empty':
            full = False
    return full

def read_ponilines(line):

    param = line.lstrip().rstrip().split(':')[0]
    if not param in ['Detector','SplineFile']:
        value = float(line.lstrip().rstrip().split(':')[1])
    else:
        value = line.lstrip().rstrip().split(':')[1]

    return {param:value}

def read_poni(filename):

    f = open(filename,'r')
    ponilines = f.readlines()

    ponidict = initiate_ponidict()

    i = 0
    while not ponidict_full(ponidict):
        ponidict.update(read_ponilines(ponilines[::-1][i]))
        i+=1
       
    return ponidict


def write_poni(ponidict,filename):

    ponilines = []
    parameterlist = ['Detector','PixelSize1','PixelSize2','Distance','Poni1','Poni2','Rot1','Rot2','Rot3','Wavelength']
    localtime = str(time.asctime( time.localtime(time.time()) ))
    ponilines.append('# \n')
    ponilines.append('# poni for troi calculated at '+localtime+'\n')
    for item in parameterlist:
        try:
            ponilines.append(item + ": " + str(ponidict[item])+ "\n")
        except ValueError:
            pass
        
    f = open(filename,'w')
    
    f.writelines(['%s' % l for l in ponilines])
    f.close()

def poni_for_troi(filename,troi=((0, 0), (2165, 2070)),troiname = 'troi1', troipath=None, rebin=None, verbose = False):
    '''
    creates a new PONI file that is corrected for the given troi\n
    This is Eiger4M specific!\n
    savefilename = filename[:,filename.find('.poni')] + '_troi%s.poni' %s troi
    '''

    ponidict = read_poni(filename)


    ponidict['Poni1']+= - troi[0][0] * ponidict['PixelSize1']

    ponidict['Poni2']+= - troi[0][1] * ponidict['PixelSize2']

    if type(rebin) != type(None):
        ponidict['PixelSize1'] = float(ponidict['PixelSize1'])*rebin[0]
        ponidict['PixelSize2'] = float(ponidict['PixelSize2'])*rebin[1]
        
    if verbose:
        print('new rebinned pixel size:')
        print(ponidict['PixelSize1'])
        print(ponidict['PixelSize1'])
    
    
    if troipath == None:
        savefilename = filename[:filename.find('.poni')] + '_%s.poni' % troiname
    else:
        savefilename = troipath + os.path.sep +  '_%s.poni' % troiname
        write_poni(ponidict,savefilename)
    return (ponidict, savefilename)


    
def main(filenames):
    for filename in filenames:
        troi = ((1174, 331), (692, 332))
        poni_for_troi(filename, troi = troi)


                 
                 
if __name__ == '__main__':
    
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc. 
\n2) python <thisfile.py> -f <file containing args as lines> 
\n3) find <*yoursearch* -> arg1 etc.> | python <thisfile.py> 
"""

    args = []
    if len(sys.argv) > 1:
        if sys.argv[1].find("-f")!= -1:
            f = open(sys.argv[2]) 
            for line in f:
                args.append(line.rstrip())
        else:
            args=sys.argv[1:]
    else:
        f = sys.stdin
        for line in f:
            args.append(line.rstrip())
    
#    print args
    main(args)
