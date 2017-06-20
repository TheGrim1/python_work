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

def test():

    print('troi = ')
    print(((20,60),(100,200)))
    print('troi_to_xy(((20,60),(100,200))) = ')
    print(troi_to_xy(((20,60),(100,200))))
    print('xy_to_troi(troi_to_xy(((20,60),(100,200)))) = ')
    print(xy_to_troi(troi_to_xy(((20,60),(100,200)))))
