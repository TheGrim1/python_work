import numpy as np

def draw_lines_troi(troi = ((0,0),(10,10)), color = "black", axes = None, linewidth = 1):
    """
    draw a box into an Eiger sized array plotted into axes = axes around the give troi = troi, with color = color.
    """
    (y,x,dy,dx) = (troi[0][0],troi[0][1],troi[1][0],troi[1][1])
 
    axes.plot((x,x+dx), (y,y) ,color = color, linewidth = linewidth)
    axes.plot((x,x+dx), (y+dy,y+dy) ,color = color, linewidth = linewidth)
    axes.plot((x+dx,x+dx), (y,y+dy) ,color = color, linewidth = linewidth)
    axes.plot((x,x), (y,y+dy) ,color = color, linewidth = linewidth)
    
    return axes

def get_vcolor(data,
               low=10,
               high=90):
    '''returns the low(10) and high(90) percentile for nice colorscaling
    '''
    vmin=(np.percentile(data,low))
    vmax=(np.percentile(data,high))
    return (vmin,vmax)

def factorize(total, more_rows=False):
    '''
    factorizes total into a number of rows an columns which will suffice to plot total number of frames in one figure
    '''
    i = np.int(np.round(np.sqrt(total)))

    if more_rows :
        return (i,i+1)
    else:
        return (i+1,i)
