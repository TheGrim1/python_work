from __future__ import print_function
import numpy as np

def rebin(a, shape):
    '''
    throws away values outside of a.shape % shape
    '''
    len0 = (a.shape[0]//shape[0])
    len1 = (a.shape[1]//shape[1])
    b = a[0: len0*shape[0], 0: len1*shape[1]]
    sh = len0,shape[0],len1,shape[1]
    return b.reshape(sh).sum(-1).sum(1)

def make_troi(coord=[0,0], size=20):
    return ((coord[0]-int(0.5*size),coord[1]-int(0.5*size)),(size,size))


def troi_to_range(troi):
    (ystart, yend, xstart, xend) = troi_to_xy(troi)

    return (range(ystart, yend), range(xstart, xend))

def troi_to_xy(troi):
    
    xstart = troi[0][1]
    xend   = troi[0][1] + troi[1][1]
    ystart = troi[0][0]
    yend   = troi[0][0] + troi[1][0]

    return tuple(ystart, yend, xstart, xend)

def xy_to_troi(ystart,yend = None, xstart = None, xend = None):

    if yend == None:
        if len(ystart)==4:
            xend   =  ystart[3]
            xstart = ystart[2]
            yend   =  ystart[1]
            ystart =  ystart[0]
        else:
            raise KeyError('not a valid troi!')
    
    troi       = [[0,0],[0,0]]
    troi[0][1] = xstart
    troi[1][1] = xend - xstart
    troi[0][0] = ystart
    troi[1][0] = yend - ystart
    
    return troi

def troi_to_slice(troi):
    
    xstart = troi[0][1]
    xend   = troi[0][1] + troi[1][1]
    ystart = troi[0][0]
    yend   = troi[0][0] + troi[1][0]


    return (slice(ystart,yend,1),slice(xstart,xend,1))

def xy_to_corners(xy):
    return troi_to_corners(xy_to_troi(xy))

def troi_to_corners(troi):
    corners_list=[]
    corners_list.append([troi[0][0],troi[0][1]])
    corners_list.append([troi[0][0] + troi[1][0],troi[0][1]])
    corners_list.append([troi[0][0],troi[0][1] + troi[1][1]])
    corners_list.append([troi[0][0] + troi[1][0],troi[0][1] + troi[1][1]])
    
    return corners_list

def array_as_list(array):
    '''
    resorts array into [[x0,y0,z0],....]
    returns shape = (3,n x m) for array,shape = (n,m)
    '''
    xyz = []
    x = np.arange(array.shape[0])
    y = np.arange(array.shape[1])

    for i in x:
        for j in y:
            xyz.append([i,j,array[i,j]])

    xyz = np.rollaxis(np.asarray(xyz),-1)
    
    

    return xyz


def check_in_border(vals_list, min_max_list):
    '''
    min max is a list of [min, max]
    '''

    for val, min_max  in zip(vals_list, min_max_list):
        if val < min_max[0]:
            return False
        if val > min_max[1]:
            return False
    return True
    


def test():

    print('troi = ')
    print(((20,60),(100,200)))
    print('troi_to_xy(((20,60),(100,200))) = ')
    print(troi_to_xy(((20,60),(100,200))))
    print('xy_to_troi(troi_to_xy(((20,60),(100,200)))) = ')
    print(xy_to_troi(troi_to_xy(((20,60),(100,200)))))


    
def goodmesh_example(xlen, ylen):
    
    y = np.arange(ylen)
    x = np.arange(xlen)
    # get shape = (xlen, ylen) result:
    # test = x**2 + np.matrix(y**2).T
    
    # get shape = (xlen, ylen) meshgrid
    xx, yy = np.meshgrid(x,y)
    # also possible : xx,yy,zz etc.
    # indz = np.array([0,3])[np.newaxis,:,np.newaxis]
    # indx = np.arange(4,6)[:,np.newaxis,np.newaxis]
    # indy = np.arange(0,2)[np.newaxis,:,np.newaxis]
    # bla = [ind[indx,indy,indz] for ind in [xx,yy,zz]]

    # get shape = (xlen, ylen) with mixed contributions:
    # test = x**2 + np.matrix(y**2).T + 2*xx - yy**2
    
    return np.array(xx,yy)


def mask_troi(data, troi,verbose=False):
    '''
    troi = ((min_val1, min_val2, ... ntimes),((max_val1-minval1), .. ntimes))
    data.shape = (arb, n) 
    troi.shape = (2,n)
    checks n values of data whether they are > troi[0][ni]
    and > troi[1][ni]
    returns a mask where this is true
    i.e the mask shows points whose n values are in the nD cube defined by troi
    return.ndim = data.ndim -1
    '''
    troi=np.asarray(troi)
    n = troi.shape[1]
    data = np.rollaxis(data,-1)
    mask = np.zeros(shape=data.shape)
    for i in range(n):
        minval = troi[0][i]
        maxval = minval + troi[1][i]
        if verbose:
            print('minval, maxval {}'.format(i))
            print(minval, maxval)
        mask[i] = np.where(data[i]>minval,1,0)
        mask[i] = np.where(data[i]>maxval,0,mask[i])

    result=np.where(mask.sum(axis=0)==n,1,0)
    if verbose:
        print('found {} points in troi'.format(result.sum()))
    return result
        
              
