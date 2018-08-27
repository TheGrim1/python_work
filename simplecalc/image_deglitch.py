import numpy as np
from scipy.ndimage import correlate1d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndshift
import warnings
import matplotlib.pyplot as plt

import sys
sys.path.append('/data/id13/inhouse2/AJ/skript')
from simplecalc.fitting import do_gauss_fit
from simplecalc.fitting import do_logistic_fit




def shift_lines(p, image):
    p = list(p)
    for i in range(len(p)):
        ndshift(image[i], p[i], output=image[i])
    return image
    
def image_error_func(p, image, reference):
    return np.ravel(shift_lines(p, image) - reference)

def homogenize_imagestack_lines_contrast(imagestack, reference_percentile=80):
    '''
    imagestack.shape = (n, height, width,...)
    loops over n and height
    scales values of the remaining dimensions to that their max an min correspond to that of the reference percentile
    '''
    reference = np.percentile(imagestack,reference_percentile,axis=0)

    for i,image in enumerate(imagestack):
        for j,line in enumerate(reference):
            min_ref = line.min()
            scale_ref = line.max()-min_ref
            min_img = image[j].min()
            scale_img = image[j].max()-min_img

            if scale_img*scale_ref != 0:
                imagestack[i][j]=((image[j] - min_img)/(scale_img))*scale_ref+min_ref
            else:
                imagestack[i][j] *= 0
                
    return imagestack


def find_edge(line,verbose=False):
    '''
    aligns on to an edge found (there should only be one!)
    edge = position where SG-filtered line crosses 0.5
    '''
    data = np.copy(line)
    data = gaussian_filter(data,sigma=2)
    first = data[0]
    last = data[-1]
    max_val = data.max()
    min_val = data.min()
    max_i = np.argmax(data)
    min_i = np.argmin(data)
    if first<last:
        data[max_i:]=max_val
        data[:min_i+1]=min_val
    else:
        data[:max_i+1]=max_val
        data[min_i:]=min_val
        data=max_val-data

    print(max_val,min_val,max_i,min_i)
        

    data_2d = np.asarray(zip(np.arange(len(data)),data))

    max_v,min_v,edge_pos,sigma = do_logistic_fit(data_2d)

    if verbose:
        plt.plot(data)
        plt.vlines(edge_pos,min_val,max_val)
        plt.show()
    
    return edge_pos

def imagestack_correct_lines_edge(imagestack, reference_percentile=80, reference_frame_no=None , verbose = False):
    '''
    corrects the lines in images in imagestack(no_images, height, width)
    along axis 2 so that they best fit with the reference_percentile of imagestack (or frame given by reference_frame_no)
    aligns on to an edge found (there should only be one!
    edge = position of the gaussian fit to the derivative
    i.e data keeps its coordinate in the axes 0 and 1 and is shifted to best agree with reference percentile in the remaining axis.
    '''
    
    data = np.copy(imagestack)


    if type(reference_frame_no)==type(None):
        reference = np.percentile(imagestack,reference_percentile,axis=0)
    else:
        referecne = np.copy(imagestack[reference_frame_no])
    
    shift_list=[]

    ref_edge_list = []
    for j, ref_line in enumerate(reference):
        ref_edge_list.append(find_edge(ref_line))

    for i, image in enumerate(data):

        if verbose:
            print('shifting image {}'.format(i))

        shift = []
        for j, ref_line in enumerate(image):
            img_line = image[j]

            ref_edge_position = ref_edge_list[j]
            img_edge_position = find_edge(img_line)
            j_shift = img_edge_position-ref_edge_position
            shift.append(j_shift)
            ndshift(data[i][j],j_shift,output=data[i][j])

            
        shift_list.append(shift)

        if verbose:
            print('found shift:')
            print(p)


    
    return (data, shift_list)



def imagestack_correlate_lines(imagestack, reference_percentile=80, reference_frame_no=None , homogenize_contrast=True, verbose = False):
    '''
    corrects the lines (planes...) in images in imagestack(no_images, height, width ...)
    along axis 2++ so that they best fit with the reference_percentile of imagestack (or frame given by reference_frame_no)
    i.e data keeps its coordinate in the axes 0 and 1 and is shifted to best agree with reference percentile in the remaining axes.
    '''


    data = np.copy(imagestack)

    if homogenize_contrast==True:
        data = homogenize_imagestack_lines_contrast(imagestack, reference_percentile)

    if type(reference_frame_no)==type(None):
        reference = np.percentile(imagestack,reference_percentile,axis=0)
    else:
        referecne = np.copy(imagestack[reference_frame_no])
    
    shift_list=[]

    for i, image in enumerate(data):

        if verbose:
            print('shifting image {}'.format(i))

        shift = []
        for j, ref_line in enumerate(reference):
            img_line = image[j]
            # smoothing relevant edges:

            mask = np.where(img_line>0,1,0)
            mask = np.where(ref_line>0,1,mask)

            smooth = np.ones(shape=(mask.sum()))
            for i in range(5):
                smooth[i]*=(i+1)/5
                smooth[-i]*=(i+1)/5
            img_line2=img_line[np.where(mask)]*smooth
            ref_line2=img_line[np.where(mask)]*smooth
            
            
            
            # smoothing noise:
            img_line2 = gaussian_filter(img_line, sigma=2, truncate=3)
            
            correlation = correlate1d(img_line2, ref_line2, axis=0, mode='constant')
            maxcorrelation = np.argmax(correlation)

            # refine with gaussian fit to maximum:
            try:
                gauss_data = np.asarray(zip(np.arange(11),correlation[maxcorrelation-5:maxcorrelation+6]))
                amp, gaussmax, sig = do_gauss_fit(gauss_data, verbose = False)
                maxcorrelation += gaussmax - 5
                
            except IndexError:
                print('index error, passing')
                pass
            
            j_shift = 0.5*len(correlation) - maxcorrelation
            shift.append(j_shift)
            ndshift(data[i][j],j_shift,output=data[i][j])

            
        shift_list.append(shift)

        if verbose:
            print('found shift:')
            print(p)

    return (data, shift_list)

def imgagestack_shift_lines(imagestack, shift_list, in_place=False):

    if not in_place:
        data=np.copy(imagestack)
    else:
        data = imagestack
        
    for i, (l_shift, image) in enumerate(zip(shift_list,data)):

        for j, (shift, line) in enumerate(zip(l_shift,image)):
            ndshift(data[i][j],shift,output=data[i][j])

    return(data)

def data_stack_shift(data, shift, lines_shift):
    '''
    arbitrary shape > 2
    idea:
    shift.shape <= data.shape
    lines_shift.shape < shift.shape
    always shifts first axes, first shift, then lines_shift
    '''

    # had weird results after ndshift if data in and data out were the same object!
    shifted_data=np.zeros_like(data)
    ndshift(data, shift=list(shift)+[0]*(data.ndim-len(shift)), output=shifted_data, order=1)
    data=np.copy(shifted_data)
    shifted_data=np.zeros_like(data)
    if type(lines_shift)!=type(None):
        for i, map_lines in enumerate(data):
            line_shift = lines_shift[i]
            if line_shift!=0:
                ndshift(map_lines, [line_shift]+[0]*(map_lines.ndim-len(line_shift)), output=shifted_data[i], order=1)
            else:
                shifted_data[i] = data[i]

    return shifted_data
