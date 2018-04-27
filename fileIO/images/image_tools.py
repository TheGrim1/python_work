
from __future__ import print_function
from __future__ import division


import shlex
import subprocess
import numpy as np
import fabio
import PIL.Image as Image
import os

from PyQt4.QtGui import QImage, qRgb

def imagefile_to_array(imagefname):
    """
    Loads image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(imagefname) as image:         
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        rows   = image.size[1]
        cols   = image.size[0]
        no_channels = int(len(im_arr)/rows/cols)
        im_arr = im_arr.reshape((rows, cols, no_channels))
        im_arr = np.rollaxis(im_arr,-1)
    return im_arr

def array_to_imagefile(data, imagefname,verbose=False):
    """
    gets a 3D Numpy array of shape 
    (width, height, channels)
    and saves it into imagefname
    """
    if data.ndim == 2:
        data = np.dstack([data,data,data])
        data = np.rollaxis(data,-1)
        # print(data.shape)
    img = Image.fromarray(np.uint8(np.rollaxis(np.rollaxis(data,-1),-1)))
        
    img.convert("RGB")
    img.mode   = "RGB"
    if verbose:
        print("saving ", os.path.realpath(imagefname))
    img.save(imagefname)
    return 1


def edf_to_image(edf_fname, image_fname):
    data = fabio.open(edf_fname).data
    data = optimize_greyscale(data)
    array_to_imagefile(data, image_fname)


def optimize_greyscale(data_in, perc_low=1, perc_high = 99,
                       out_max = 255, dtype = "uint8", mask = None):
    '''
    optimizes the scaling of data to facilitate alignment procedures
    inverts scale if at percentile (<prec_low> + <perc_high>)/2 the luninosity is less* than 0.5
    * swithched with <foreground_is_majority>
    default is change dytpe to: "int8" else pass None
    '''
    if type(mask)==type(None):
        low  = np.percentile(data_in,perc_low)
        high = np.percentile(data_in,perc_high)
    else:
        low  = np.percentile(np.where(mask,0,data_in),perc_low)
        high = np.percentile(np.where(mask,0,data_in),perc_high)
        
    #print '0-',low,high
    data = np.copy(data_in)
    data = data*1.0 # floatify
    data = (data - low)/(high-low)

    data = np.where(data<0,   0.0,data)
    data = np.where(data>1.0, 1.0,data)

    optimized = data * out_max
    
    if dtype == "uint8":
        optimized = np.asarray(np.where(optimized<0,   0,optimized), dtype = np.uint8)
        
    return optimized


def open_series(find_path,find_arg, verbose = False):
    '''
    opens a series of images found with unix: find <prefix>
    sorts the found file names
    '''
    
    arg       = []
    arg.append("find")
    arg.append(find_path)
    arg.append('-name')
    arg.append(find_arg)

    all_fnames =shlex.split(subprocess.check_output(arg))
    all_fnames.sort()
    #all_fnames = all_fnames[1::]
    
    image_list = []
    for i, fname in enumerate(all_fnames):
        image_list.append(imagefile_to_array(fname))
        if verbose:
            print('opening image ' , fname)
    imagestack = np.stack(image_list)
    return imagestack

def save_series(data, savename="default.png", savename_list = None, verbose=True):
    '''
    saves 3d or 4d array as imagesfiles: 
         3d (imagenumber, width, height)
         4d (imagenumber, width, height, channels)
    if savename_list is a list with the correct length these are used to save images, 
    else savename is appended with _X (before the suffix)
    '''
    if type(savename_list) == list:
        if len(savename_list) == data.shape[0]:
            if verbose:
                print('using given names: ')
                print(savename_list)
        else:
            raise ValueError('incorrect savename_list len : ', len(savename_list))
    else:
        digits = len(str(data.shape[0]))
        save_path = os.path.dirname(savename)
        save_prefix = os.path.basename(savename.split('.')[:-2])
        save_suffix = os.path.basename(savename.split('.')[-1])
        savename_list = [os.path.sep.join([save_path,save_prefix])+str(i).zfill(digits) + save_suffix for i in range(data.shape[0])]
        if verbose:
            print('made names list:')
            print(save_name_list)
            
    for i, image in enumerate(data):
        array_to_imagefile(image, savename_list[i],verbose=verbose)
            
            
    return True





def uint8array_to_qimage(im, copy=False):
    '''from https://gist.github.com/smex/5287589
    '''
    gray_color_table = [qRgb(i, i, i) for i in range(256)]

    if im is None:
        return QImage()

    if im.dtype == np.uint8:
        if len(im.shape) == 2:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                return qim.copy() if copy else qim
