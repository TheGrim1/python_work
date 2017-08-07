
import numpy as np
import PIL.Image as Image

def imagefile_to_array(imagefname):
    """
    Loads image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(imagefname) as image:         
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        rows   = image.size[1]
        cols   = image.size[0]
        no_channels = len(im_arr)/rows/cols
        im_arr = im_arr.reshape((rows, cols, no_channels))
        im_arr = np.rollaxis(im_arr,-1)
    return im_arr

def array_to_imagefile(data, imagefname):
    """
    gets a 3D Numpy array of shape 
    (width, height, channels)
    and saves it into imagefname
    """
    img = Image.fromarray(np.rollaxis(np.rollaxis(data,-1),-1))
    img.save(imagefname)
    return 1


def optimize_greyscale(data_in, perc_low=10, perc_high = 90, foreground_is_majority = True, out_max = 255):
    '''
    optimizes the scaling of data to facilitate alignment procedures
    inverts scale if at percentile (<prec_low> + <perc_high>)/2 the luninosity is less* than 0.5
    * swithched with <foreground_is_majority>
    '''
    low  = np.percentile(data_in,perc_low)
    high = np.percentile(data_in,perc_high)
    print low,high
    data = np.copy(data_in)
    data = (data - low) / (high-low)
    
    data = np.where(data<0, 0.0,data)
    data = np.where(data>1.0, 1,data)

    
    if np.percentile(data,50) < 0.5:
        data = 1.0 - data
        

    
    optimized = data * out_max
    return optimized