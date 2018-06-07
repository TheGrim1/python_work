from __future__ import print_function
from __future__ import division

import numpy as np
import scipy.ndimage as nd
#from silx.image import sift
import timeit
import time
import sys, os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# local imports
path_list = os.path.dirname(__file__).split(os.path.sep)
importpath_list = []
if 'skript' in path_list:
    for folder in path_list:
        importpath_list.append(folder)
        if folder == 'skript':
            break
importpath = os.path.sep.join(importpath_list)
sys.path.append(importpath)        

from simplecalc.slicing import troi_to_slice
from simplecalc.gauss_fitting import fit_2d_gauss
from simplecalc.rotated_series import rotate_series
from simplecalc.gauss_fitting import do_multi_gauss_fit
from userIO.GenericIndexTracker import run_GenericIndexTracker
from fileIO.datafiles import open_data

def do_shift(imagestack, shift):
    shift = list(shift)
    for i in range(imagestack.shape[0]):
        ishift = shift[i]
        nd.shift(imagestack[i],ishift, output = imagestack[i])    
    return imagestack
    

def centerofmass_align(imagestack, alignment= None):
    '''
    shifts the images in imagestack so that their center of mass lies on the COM of imagestack[0]
    works for arbitrary dimensions
    only aligns for axes where alignmetn != 0, defaults to ones
    '''
    if type(alignment) == type(None):
        alignment = np.ones_like(imagestack.shape)
    COM   = []
    shift = []
    for i in range(imagestack.shape[0]):
        COM.append(np.asarray(nd.measurements.center_of_mass(imagestack[i])))

        ishift = 0.0*np.array(COM[0])
        for val_no, val in enumerate(COM[i]):
            if alignment[val_no]:
                ishift[val_no]= COM[0][val_no] - val

        shift.append(ishift)
        #print shift
        nd.shift(imagestack[i],ishift, output = imagestack[i])
    

    shift = np.asarray(shift)
    
    return imagestack, shift



def userclick_align(imagestack, norm='linear', coordinates_fname=None):
    '''
    prompts user input for each frame aligning them to the first frame and selected pixel
    '''



    # if coordinates_fname==None:
    #     coordinates_fname = os.getcwd() + os.path.sep + "image_align_coordinates_{}.dat".format(int(time.time()))
    # else:
    #     if not os.path.exists(os.path.dirname(coordinates_fname)):
    #         raise ValueError('path doe not exist accessible {}'.format(os.path.dirname(coordinates_fname)))
    #     elif os.path.exists(coordinates_fname):
    #         print('deleting previous coordinate file {}'.format(coordinates_file))
    #         os.remove(coordinates_fname)
                              
    arg_list = [[imagestack, norm, coordinates_fname]]

    coords = run_GenericIndexTracker(arg_list[0])

    shift = np.asarray([[[coords[0,1] - x[1],coords[0,0]-x[0]] for x in coords]][0])

    print(coords)
    print(shift)
    
    for i in range(imagestack.shape[0]):
        ishift = shift[i]
        # print(ishift)
        nd.shift(imagestack[i],ishift, output = imagestack[i])
        
    shift=np.asarray(shift)

    return imagestack, shift


def crosscorrelation_align_1d(imagestack, axis = 1):
    ''' 
    forms the 1d cross correlation of all images with imagestack[0,:,:]
    along <axis> =                                                 0,1
    shifts imagestack[n,:,:] to to the maximum
    less memory intense than full CC-align
    '''
    shift=[]
    reference = np.copy(imagestack[0])
    for i in range(imagestack.shape[0]):
        print('aligning 0 with %s' %i)

        (imagestack[i], ishift) = single_correlation_align_1d(reference,
                                                              imagestack[i],
                                                              axis=axis)
        shift.append(ishift)
        
    shift=np.asarray(shift)

    return imagestack, shift


def crosscorrelation_align(imagestack):
    '''forms the cross correlation of all images with imagestack[0,:,:]
    and shifts them to to the maximum
    memory intense use small ROIs!
    '''
    shift=[]
    reference = np.copy(imagestack[0])
    for i in range(imagestack.shape[0]):
        print('aligning 0 with %s' %i)

        (imagestack[i], ishift) = single_correlation_align(reference,
                                                           imagestack[i])
        shift.append(ishift) 
    shift=np.asarray(shift)
    return imagestack, shift

def forcedcrosscorrelation_align(imagestack, alignment = (0,0)):
    '''forms the correlation between imagestack[0] and all others. The shift
is center of refernce - max of the crosscorrelation. Aligns imagestack
to imagestack[0,:,:].\n Applies optional linear potential from
max(alingimagestack[0,:,:]) to min(imagestack[0,:,:]) in direction
alignment (see mask_align). \n tackes 3d array (X, N, M) and alligns
all in X \nreturn (imagestack, shift).\nAs the image is forced towards
the direction given in alignment, repeated runs may continue to shift
the image!'''
    shift = []

#    setup the potential to force the alignment in a certain direction
    maxforce = imagestack[0,:,:].max()/10.0
    minforce = imagestack[0,:,:].min()/10.0
    xstepforce = (maxforce + minforce)/ len(imagestack[0,:,0])
    ystepforce = (maxforce + minforce)/ len(imagestack[0,0,:])

    print('alignment = ')
    print(alignment)
    if alignment[0]:
        x = np.atleast_2d(np.arange(minforce, maxforce, xstepforce)).T        
        x = x[::-alignment[0]]
    else:
        x = np.atleast_2d(np.ones(imagestack.shape[1])).T

    if alignment[1]:
        y = np.atleast_2d(np.arange(minforce, maxforce, ystepforce))
        y = y[0][::-alignment[1]]
    else:
        y = np.atleast_2d(np.ones(imagestack.shape[2]))
    
    if any(alignment):
        potential = x * y
        print('potential.shape = ')
        print(potential.shape)
        reference = np.copy(imagestack[0])
    else:
        reference = np.copy(imagestack[0])
        potential = x * y
        

# run over all including reference to apply the forcing evenly, repeated runs may continue to shift the image!
    for i in range(imagestack.shape[0]):
        print('aligning 0 with %s' %i)
        
        forcedalign = np.copy(imagestack[i]) * potential

        (forcedalign,ishift) = single_correlation_align(potential * reference, forcedalign)
        nd.shift(imagestack[i],ishift,output = imagestack[i])
        shift.append(ishift)                                  

# plot residue to reference
#        imagestack[i,:,:] = reference - imagestack[i,:,:] 

#        imagestack[i,:,:] = reference
        
    shift = np.asarray(shift)
    return imagestack, shift

def correlation_task(args):
    reference = args[0]
    image     = args[1]
    shift     = args[2]
    i         = args[3]
    reflen    = args[4]
    return (reference*nd.shift(image,shift*(i - reflen))).sum()

def single_correlation_alirgn_1d(reference, image, axis = 1):
    '''
    cross correlation between reference and image along <axis>
    shifts image to the maximum
    returns (image, shift)
    '''


    PROCESSES = cpu_count()
    print('cpu_count() = %d\n' % PROCESSES)

    #
    # Create pool
    #


    print('Creating pool with %d processes\n' % PROCESSES)
    pool = Pool(PROCESSES)

    shift       = np.zeros(reference.ndim)
    shift[axis] = 1.0
    reflen = reference.shape[axis]
    
    task_list = [(reference, image, shift, i, reflen) for i in range(2*reflen)]
    correlation = pool.map(correlation_task, task_list)

    pool.close()

    maxcorrelation = np.argmax(correlation)


    ### sub-pixel refining with gaussian fit:
    ### there seems to be a problem wiht do_multi_gauss_fit :(
    # data = np.asarray([np.arange(11),correlation[maxcorrelation-5:maxcorrelation+6]]) 

    # amp, gaussmax, sig = do_multi_gauss_fit(data, nopeaks = 1, verbose = True)
    
    # maxcorrelation += gausmax - 5

    
    # print 'maxcorrelation after gauss: ', maxcorrelation
    
    shift          = shift*(maxcorrelation - reference.shape[axis])
    
    nd.shift(image,shift, output = image)
    shift = np.asarray(shift)
    return (image, shift)

def single_correlation_align(reference, image):
    'see crosscorrelation before stacks were cool'

    correlation = nd.correlate(image, reference, mode = 'constant')
    maxcorrelation = np.argmax(correlation)

    maxcorrelation = np.array(np.unravel_index(maxcorrelation,correlation.shape))

    
    ## refine on sub pixel level with gauss fit

    # print 'before refinement: ' , maxcorrelation 
    area           = np.array((10,10))
    peak_troi      = (np.array(maxcorrelation) - int(area/2.0), area)


    fitting_region = np.array(correlation[troi_to_slice(peak_troi)])

    (amp, x0, y0, a,b,c,), residual = fit_2d_gauss(fitting_region)
    
    # import fileIO.plots.plot_array as pa
    # plottable = fitting_region
    # pa.plot_array(fitting_region)

    # print 'peaktroi = ',peak_troi
    # print 'fitresult x,y = ', x0, ',' , y0
    
    maxcorrelation = peak_troi[0] + np.array((x0,y0))
    # print 'after refinement: ' , maxcorrelation
    # print 'gauss residual =: ' , residual
    
    centerref      = 0.5*np.array(reference.shape)

    shift = centerref - maxcorrelation

#    print "maxcorrelation = %s" %maxcorrelation 
#    print "centerref = " 
#    print centerref 
#    print 'reference.shape = ',reference.shape
#    print 'image.shape = ',image.shape
#    print 'shift = ',shift
#
#    print 'reference:'
#    plt.matshow(reference)
#    print 'image:'
#    plt.matshow(image)
#    print 'correlation'
#    plt.matshow(correlation)
#        

    nd.shift(image,shift, output = image)
    shift = np.asarray(shift)
    return (image, shift)
#    return (correlation, shift)


def mask_align(imagestack, threshold = 5, alignment = (0,0)):
    'masks the aling array by thresholding.\n Aligns to the alighnment = (1,1) = top - left/n(-1,1) = btm - left/n(1,-1) = top - right/n(-1,-1) = bottom - right corner. A value of 0 in alignment does not align that direction.\n Returns the imagestack array aligned to the first array in the stack and a list of the shift (r[0]-a[0],r[1]-a[1]).'

    
    maskreference = np.where(imagestack[0,:,:] > threshold, 1, 0)
    
    shift = [(0,0)]
    drefcols  = 0

# count 0s for reference  
    if alignment[0]:
        refcols   = maskreference.sum(1)[::alignment[0]]
        i = 0
        while not refcols[i]:        
            drefcols += 1
            i    += 1
    drefrows = 0
    if alignment[1]:
        refrows  = maskreference.sum(0)[::alignment[1]]   
        i        = 0
        while not refrows[i]:        
            drefrows += 1
            i        += 1

    # print 'found drefcols = %s and drefrows = %s' %(drefcols,drefrows)
    
    for k in range(1,imagestack.shape[0]):
        kshift = [0,0]
        maskalign     = np.where(imagestack[k,:,:] > threshold, 1, 0)
        if alignment[0]:
            aligncols = maskalign.sum(1)[::alignment[0]]
            dcols  = 0
            i = 0
            while not aligncols[i]:        
                dcols += 1
                i    += 1
            kshift[0] = (dcols - drefcols)*alignment[0]*(-1)
        if alignment[1]:
            alignrows = maskalign.sum(0)[::alignment[1]]
            drows  = 0
            i = 0
            while not alignrows[i]:        
                drows += 1
                i     += 1
                kshift[1] = (drows - drefrows)*alignment[1]*(-1)

        shift.append(kshift)
        imagestack[k,:,:] = nd.shift(imagestack[k,:,:],shift[k])

    shift = np.asarray(shift)
    return (imagestack, shift)


# def sift_align(imagestack, threshold = None):
#     'uses the silx.image.sift alignment. Returns aligned array and tupel  (r[0]-a[0],r[1]-a[1]). The optional threshold has not been shown to help for single wire images.'

# # initialize sift according to mail from Pierre Paleo

#     reference = np.ascontiguousarray(np.where(imagestack[0,:,:] > threshold, imagestack[0,:,:], 0))

#     alignplan = sift.LinearAlign(reference, devicetype="GPU")
# # alternative:
# #    alignplan = sift.LinearAlign(reference, device=(2,0))
    
#     shift = [(0,0)]
#     for i in range(1,imagestack.shape[0]):
        
#         dummy = np.copy(imagestack[i,:,:])
#         dummy = np.where(dummy > threshold, dummy, 0)

#         result  = alignplan.imagestack(dummy, shift_only = True, return_all=True)
#         if result:
#             shift.append(np.round(np.asarray(result['offset'],dtype = float)))
#         else:
#             shift.append(np.zeros(shape=2))

#         imagestack[:,:,i] = nd.shift(imagestack[i,:,:],shift[i])


#    shift = np.asarray(shift)
#     return (imagestack, shift)

def shift_image(data, shift):
    for i, ishift in enumerate(shift):
        nd.shift(data[i], ishift, output = data[i],)
    
    return data

def real_from_rel(frame,data,shift = [1,1]):
    'this function gives you the real frame number if you give it the shifted frame number, an examplary dataset and the shift.\n returns -1 if the frame is out of the FOV.\n accepts slices'

    shift[0] = -shift[0]
    shift[1] = -shift[1]
    
    frames = np.arange(data.size)
    print(frames)
#    print shift
    frames = frames.reshape(data.shape)
    print(frames)
    frames = nd.shift(frames,shift,cval = -1)
    print(frames)
    frames = frames.flatten()
    print(frames[frame])
    
    return frames[frame]

def image_align(imagestack, mode = {'mode':'sift'}):
    '''
    returns the imagestack array stack aligned to the r = reference = imagestack[:,:,0] in it. 
    The relative shift ist areturned as a list of touples (r[0]-a[0],r[1]-a[1])
    wrapper for the other functions in this file:
    mask, crosscorrelation, forcedcrosscorrelation, centerofmass, elastix, crosscorrelation_1d, userclick
    '''

    shift = []
    
    # if mode['mode'] == 'sift':
    #     (imagestack, shift) = sift_align(imagestack, threshold = mode['threshold'])
    if mode['mode'] == 'mask':
        (imagestack, shift) = mask_align(imagestack, threshold = mode['threshold'], alignment = mode['alignment'])
    elif mode['mode'] == 'userclick':
        (imagestack, shift) = userclick_align(imagestack) 
    elif mode['mode'] == 'crosscorrelation':
        (imagestack, shift) = crosscorrelation_align(imagestack)
    elif mode['mode'] == 'forcedcrosscorrelation':
        (imagestack, shift) = forcedcrosscorrelation_align(imagestack, alignment = mode['alignment'])
    elif mode['mode'] == 'centerofmass':
        (imagestack, shift) = centerofmass_align(imagestack, alignment = mode['alignment'])
    elif mode['mode'] == 'elastix':
        from simplecalc.image_align_elastix import elastix_align
        (imagestack, shift) = elastix_align(imagestack, mode = mode['elastix_mode'])
    elif mode['mode'] == 'crosscorrelation_1d':
        (imagestack, shift) = crosscorrelation_align_1d(imagestack, axis = mode['axis'])
    else:
        print("%s is not a valid mode" % mode)    
        
    return (imagestack, shift)


def do_test():
    '''testfuction'''


    # local imports
    path_list = os.path.dirname(__file__).split(os.path.sep)
    importpath_list = []
    if 'skript' in path_list:
        for folder in path_list:
            importpath_list.append(folder)
            if folder == 'skript':
                break
    importpath = os.path.sep.join(importpath_list)
    sys.path.append(importpath)        
    from fileIO.plots.plot_array import plot_array
    
    # setup arrays to test alignment

    x         = np.atleast_2d(np.arange(100))
    y         = np.atleast_2d(np.arange(80)).T
    shift     = (5,15)
    reference = -((50-x)/100.0)**2 *((50-y)/100.0)**2 + 0.0625
    print('min(reference) = ',np.min(reference))
    reference[50:55,50:55] = 0
    imagestack1    = nd.shift(reference,shift)
    imagestack2    = nd.shift(imagestack1,shift)
    imagestack     = np.rollaxis(np.dstack([reference, imagestack1, imagestack2]),-1)
    print('set up test data with shift = ')
    print(((0,0),(shift[0],shift[1]),(shift[0]*2,shift[1]*2)))
    print()
    
    plot_array(imagestack, title = 'test data')
    
    #    timing:
    # start_time = timeit.default_timer()
    # (dummy, (foundshift)) = image_align(dummy, mode = {'mode':'sift'})
    # print 'sift shift found:'
    # print foundshift
    # print 'took %s' % (timeit.default_timer() - start_time)
    # dummy     = np.copy(imagestack)


  
    ### testing COM aling
    print('testing center of mass aling')
    start_time = timeit.default_timer()
    dummy     = np.copy(imagestack)
    mode = {'mode':'centerofmass','threshold':np.percentile(imagestack,70),'alignment':(1,1)}
    (dummy, (foundshift)) = image_align(dummy, mode = mode)
    print('%s shift found:' %mode['mode'])
    print(foundshift)
    print('took %s' % (timeit.default_timer() - start_time))

    plot_array(dummy, title = 'COM align')
 
    
    ### testing maskshift
    print('testing maskshift')
    start_time = timeit.default_timer()
    dummy     = np.copy(imagestack)
    mode = {'mode':'mask','threshold':np.percentile(imagestack,70),'alignment':(-1,1)}
    (dummy, (foundshift)) = image_align(dummy, mode = mode)
    print('%s shift found:' %mode['mode'])
    print(foundshift)
    print('with alignment')
    print(mode['alignment'])
    print('took %s' % (timeit.default_timer() - start_time))

    plot_array(dummy, title = 'masked align')

 ### testing correlation aling
    print('testing correlation aling')
    start_time = timeit.default_timer()
    dummy     = np.copy(imagestack)
    mode = {'mode':'crosscorrelation','threshold':np.percentile(imagestack,70)}
    (dummy, (foundshift)) = image_align(dummy, mode = mode)
    print('%s shift found:' %mode['mode'])
    print(foundshift)
    print('took %s' % (timeit.default_timer() - start_time))

    plot_array(dummy, title = 'cross correllation align')

 ### testing correlation aling 1d axis = 1
    print('testing correlation aling 1d')
    start_time = timeit.default_timer()
    dummy     = np.copy(imagestack)
    mode = {'mode':'crosscorrelation_1d','axis':1}
    (dummy, (foundshift)) = image_align(dummy, mode = mode)
    print('%s shift found:' %mode['mode'])
    print(foundshift)
    print('took %s' % (timeit.default_timer() - start_time))

    plot_array(dummy, title = 'cross correllation align 1d 1')

 ### testing correlation aling 1d axis = 0
    print('testing correlation aling 1d')
    start_time = timeit.default_timer()
    dummy     = np.copy(imagestack)
    mode = {'mode':'crosscorrelation_1d','axis':0}
    (dummy, (foundshift)) = image_align(dummy, mode = mode)
    print('%s shift found:' %mode['mode'])
    print(foundshift)
    print('took %s' % (timeit.default_timer() - start_time))

    plot_array(dummy, title = 'cross correllation align 1d 0')

    
    
    ### testing forcedcorrelation
    print('testing forced correlation')
    dummy     = np.copy(imagestack)
    start_time = timeit.default_timer()
    mode = {'mode':'forcedcrosscorrelation','alignment':(1,1)}
    (dummy, (foundshift)) = image_align(dummy, mode = mode)
    print('%s shift found:' %mode['mode'])
    print('with alignment')
    print(mode['alignment'])
    print(foundshift)
    print('took %s' % (timeit.default_timer() - start_time))
    plot_array(dummy, title = 'forced correlation')

    ### testing elastix
    print('testing elastix rigid align')
    dummy     = np.copy(imagestack)
    start_time = timeit.default_timer()
    mode = {'mode':'elastix'}
    (dummy, (foundshift)) = image_align(dummy, mode = mode)
    print('%s shift found:' %mode['mode'])
    print(foundshift)
    print('took %s' % (timeit.default_timer() - start_time))
    plot_array(dummy, title = 'elastix')

 
   
if __name__ == '__main__':

    do_test()
    
    
