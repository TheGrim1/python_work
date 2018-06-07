import numpy as np

def generate_random_color_list(no_colors, saturation=0.80,value=0.99):
    '''
    returns rgb values for each color, multiply by [0,1] to add greyscale
    '''
    color_list = []
    golden_ratio_conjugate = 0.618033988749895
    h = np.random.random() # use random start value
    for i in range(no_colors):
        h += golden_ratio_conjugate
        h %= 1
        color_list.append(hsv_to_rgb(h, saturation, value))
    return color_list

def hsv_to_rgb(h, s, v):
    '''
    hue, saturation, value
    to
    red green blue
    '''

    h_i = int(h*6)
    f = h*6 - h_i
    p = v * (1 - s)
    q = v * (1 - f*s)
    t = v * (1 - (1 - f) * s)
    if h_i==0:
        r, g, b = v, t, p
    if h_i==1:        
        r, g, b = q, v, p
    if h_i==2:
        r, g, b = p, v, t 
    if h_i==3:
        r, g, b = p, q, v
    if h_i==4:
        r, g, b = t, p, v
    if h_i==5:
        r, g, b = v, p, q 
    return [int(r*256), int(g*256), int(b*256)]

def hsv_to_rgb_array(hsv_array):
    '''
    hue, saturation, value
    to
    red green blue
    '''

    h = hsv_array[:,:,0]
    s = hsv_array[:,:,1]
    v = hsv_array[:,:,2]
    
    h_i = np.int32(h*6)
    f = h*6 - h_i
    p = v * (1 - s)
    q = v * (1 - f*s)
    t = v * (1 - (1 - f) * s)
    rgb_array = np.empty(shape=hsv_array.shape)
    rgb_array = np.where(np.dstack([h_i,h_i,h_i])==0,np.dstack([v,t,p]),rgb_array)
    rgb_array = np.where(np.dstack([h_i,h_i,h_i])==1,np.dstack([q,v,p]),rgb_array)
    rgb_array = np.where(np.dstack([h_i,h_i,h_i])==2,np.dstack([p,v,t]),rgb_array)
    rgb_array = np.where(np.dstack([h_i,h_i,h_i])==3,np.dstack([p,q,v]),rgb_array)
    rgb_array = np.where(np.dstack([h_i,h_i,h_i])==4,np.dstack([t,p,v]),rgb_array)
    rgb_array = np.where(np.dstack([h_i,h_i,h_i])==5,np.dstack([v,p,q]),rgb_array)

    return rgb_array


def color_distance(rgb1,rgb2):
    r1,g1,b1 = np.asarray(rgb1,dtype=np.int32)
    r2,g2,b2 = np.asarray(rgb2,dtype=np.int32)

    return np.sqrt((r1-r2)**2+(g1-g2)**2+(b1-b2)**2)

def min_distance_to_neighbours(i,j,color_array):
    distances = []
    k,l = np.meshgrid(range(-1,2),range(-1,2))
    for k,l in zip(k.flatten(),l.flatten()):
        if (i+k) in range(color_array.shape[0]) and (j+l) in range(color_array.shape[1]):

            distances.append(color_distance(color_array[i,j],color_array[i+k,j+l]))

    distances.sort()

    return distances[1] # [0] is self referetial

def some_color_list(number):
    colors = ['r','b','g','black','yellow','magenta','orange','purple','darkblue','darkred','darkyellow','darkgreen']
    return colors[number % len(colors)]
                       

def color_pixelarray(arrayshape, min_distance=50):
    color_array = np.zeros(shape=(arrayshape[0],arrayshape[1],3),dtype = np.uint8)
    color_list = generate_random_color_list(arrayshape[0]*arrayshape[1]*2)
    color_counter=0
    for i in range(arrayshape[0]):
        for j in range(arrayshape[1]):
            color_array[i,j] = color_list[color_counter]
            color_counter += 1
            while min_distance_to_neighbours(i,j,color_array) < min_distance:
                # print('color_counter')
                # print color_counter
                color_array[i,j] = color_list[color_counter]
                color_counter += 1

    return color_array

def colorify_stack(data, saturation = 0.99, min_percentile=1, max_percentile=90):
    '''
    data.shape = (height, width, n)
    translates axis <n> into a hue (is h,s,v)
    return image (hight, width, RGB) 
    where along axis n the RGB values are summed up weighted by data
    '''
    
    rgb_image = np.zeros(shape=list(data.shape[0:2])+[3])
    min_norm = np.percentile(data, min_percentile)
    stretch_norm = np.percentile(data, max_percentile) - min_norm
    if stretch_norm == 0.0:
        stretch_norm = 1.0
    no_frames = data.shape[2]
    
    for i in range(no_frames):
        hsv_image = np.empty(shape=rgb_image.shape)
        values = np.asarray(data[:,:,i], dtype=np.float64)-min_norm

        values = values/stretch_norm
        values = np.where(values<0,0,values)
        values = np.where(values>1.0,1.0,values)
        hsv_image[:,:,0] = float(i)/no_frames
        hsv_image[:,:,1] = saturation
        hsv_image[:,:,2] = values
        rgb_image += hsv_to_rgb_array(hsv_image)

    return rgb_image
