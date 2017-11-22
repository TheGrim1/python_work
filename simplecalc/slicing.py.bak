import numpy as np


def troi_to_xy(troi):
    
    xstart = troi[0][1]
    xend   = troi[0][1] + troi[1][1]
    ystart = troi[0][0]
    yend   = troi[0][0] + troi[1][0]

    return (ystart, yend, xstart, xend)

def xy_to_troi(ystart,yend = None, xstart = None, xend = None):

    if yend == None:
        if len(ystart)==4:
            xend   =  ystart[3]
            xstart = ystart[2]
            yend   =  ystart[1]
            ystart =  ystart[0]
        else:
            raise KeyError, 'not a valid troi!'
    
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


    return [slice(ystart,yend,1),slice(xstart,xend,1)]

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
