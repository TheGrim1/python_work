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
