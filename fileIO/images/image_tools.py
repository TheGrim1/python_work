
from __future__ import print_function
from __future__ import division


import shlex
import subprocess
import numpy as np
import fabio
import PIL.Image as Image
import os

from PyQt4.QtGui import QImage, qRgb
import cv2

def images_to_video(source_folder, save_fname=None, fps=40):
    '''
    finds_all files in a folder and writes then to a .avi viedo file
    assumes all files are images! ignores avi files
    '''

    fname_list = [os.path.join(source_folder,x) for x in os.listdir(source_folder) if x.find('.avi')<0]
    fname_list.sort()
    if type(save_fname) == type(None):
        save_fname = os.path.join(source_folder, os.path.splitext(os.path.basename(fname_list[0]))[0]+'.avi')

        frame_shape = imagefile_to_array(fname_list[0]).shape
    frame_shape=(frame_shape[2],frame_shape[1])
    print(frame_shape)
    writer = cv2.VideoWriter(save_fname, cv2.VideoWriter_fourcc(*"MJPG"), fps,frame_shape)
    for i, fname in enumerate(fname_list):
        print('recording frame {} of {}'.format(i+1, len(fname_list)))
        frame = imagefile_to_array(fname)
        writer.write(np.dstack([frame[0],frame[1],frame[2]]))
    writer.release()
    
def imagestack_to_video(imagestack, output_name, fps):
    '''
    the imagestack way is not good, just an example 
    '''
    writer = cv2.VideoWriter(output_name+".avi", cv2.VideoWriter_fourcc(*"MJPG"), fps,frame_shape)
    for frame in imagestack:
        writer.write(np.dstack[frame,frame,frame])
    writer.release()


def imagefile_to_array(imagefname):
    """
    Loads image into 3D Numpy array of shape 
    (channels, height, width)
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
    if data.ndim == 2:
        if data.shape[3] == 3:
            img = img.convert(mode="RGB")
            img.mode='RGB'
        if data.shape[3] == 4:
            img = img.convert(mode="RGBA")
            img.mode='RGBA'
        
        
    if verbose:
        print("saving ", os.path.realpath(imagefname))
    img.save(imagefname)
    return 1


def edf_to_image(edf_fname, image_fname, perc_low=0.1, perc_high = 99.9, mask=None):
    data = fabio.open(edf_fname).data
    data = optimize_greyscale(data, perc_low=perc_low, perc_high=perc_high, mask=mask)
    array_to_imagefile(data, image_fname)


def optimize_greyscale(data_in, perc_low=0.1, perc_high = 99.9,
                       out_max = 255, dtype = "uint8", mask = None):
    '''
    optimizes the scaling of data to facilitate alignment procedures
    * swithched with <foreground_is_majority>
    default is change dytpe to: "int8" else pass None
    '''
    if type(mask)==type(None):
        low  = np.percentile(data_in,perc_low)
        high = np.percentile(data_in,perc_high)
    else:
        not_mask = np.where(mask,0,1)
        if not_mask.sum()>0:
            low  = np.percentile(data_in[np.where(not_mask)],perc_low)
            high = np.percentile(data_in[np.where(not_mask)],perc_high)
        else:
            return np.zeros_like(data_in)
        
    #print '0-',low,high
    data = np.copy(data_in)
    data = np.asarray(data,dtype=np.float64)  # floatify
    data = (data - low)/(high-low)

    data = np.where(data<0.0,   0.0,data)
    data = np.where(data>1.0, 1.0,data)
    
    optimized = data * out_max
    
    if dtype == "uint8":
        optimized = np.asarray(np.where(optimized<0,   0,optimized), dtype = np.uint8)
        
    return optimized

def optimize_imagestack_contrast(imagestack, cutcontrast):
    print('optimizing image contrast with {}'.format(cutcontrast))

    imagestack_max = np.max(imagestack)

    if cutcontrast > 0:
        print('cutting low intensities')
        imagestack=np.where(imagestack<abs(cutcontrast)*imagestack_max,0,imagestack)

    else:
        print('cutting high intensities, inverting')

        imagestack = np.where(imagestack>abs(cutcontrast)*imagestack_max,imagestack_max,imagestack)
        imagestack = np.max(imagestack) - imagestack

    pixel_low = np.where(imagestack>0,0,1)
    perc_low = 0
    perc_high = 100
    imagestack = optimize_greyscale(imagestack, perc_low=perc_low, perc_high=perc_high, mask=pixel_low)


    return imagestack


def crop_imagestack(imagestack,shape,centers=None):
    '''
    crops the imagestack down to len(imagestack) x shape
    if centers is given len(centers) must = len(imagestack)
    the data[i] is cropped around centers[i] 
    else the image is cropped equally from all sides
    '''
    if type(centers)==type(None):
        centers = imagestack.shape[0]*[imagestack.shape[1],imagestack.shape[2]]
    else:
        centers = [[int(x[0]),int(x[1])] for x in centers]
        
    corners = [[x[0]-int(shape[0]/2),x[1]-(shape[1]/2)] for x in centers]

    cropped = np.zeros(shape=(len(imagestack),shape[0],shape[1]))

    for i,image in enumerate(imagestack):
        u = int(corners[i][0]+shape[0])
        d = int(corners[i][0])
        l = int(corners[i][1])
        r = int(corners[i][1]+shape[1])
        cropped[i] = image[d:u,l:r]
    
    return cropped 

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
