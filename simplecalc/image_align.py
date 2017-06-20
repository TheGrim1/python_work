import numpy as np
import scipy.ndimage as nd
#from silx.image import sift
import timeit

import matplotlib.pyplot as plt

def crosscorrelation_align(align, alignment = (0,0)):
    'forms the correlation between align[0] and all others. The shift is center of refernce - max of the crosscorrelation. Aligns align to align[0].\n Applies optional linear potential from max(aling[0]) to min(align[0]) in direction alignment (see mask_align). \n tackes 3d array (X, N, M) and alligns all in X \nreturn (align, shift).\nAs the image is forced towards the direction given in alignment, repeated runs may continue to shift the image! '
    shift = []

#    setup the potential to force the alignment in a certain direction
    maxforce = align[:,:,0].max()
    minforce = align[:,:,0].min() 
    xstepforce = (maxforce + minforce) / len(align[:,0,0])
    ystepforce = (maxforce + minforce) / len(align[0,:,0])

    print 'alignment = '
    print alignment
    if alignment[0]:
        x = np.atleast_2d(np.arange(minforce, maxforce, xstepforce)).T        
        x = x[::-alignment[0]]
    else:
        x = np.atleast_2d(np.ones(align.shape[0])).T

    if alignment[1]:
        y = np.atleast_2d(np.arange(minforce, maxforce, ystepforce))
        y = y[0][::-alignment[1]]
    else:
        y = np.atleast_2d(np.ones(align.shape[1]))
    
    if any(alignment):
        potential = x * y
        print 'potential.shape = '
        print potential.shape
        reference = np.copy(align[:,:,0])
    else:
        reference = np.copy(align[:,:,0])
        potential = x * y
        

# run over all including reference to apply the forcing evenly, repeated runs may continue to shift the image!
    for i in range(align.shape[2]):
        print 'aligning 0 with %s' %i
        
        forcedalign = np.copy(align[:,:,i]) * potential

        (forcedalign,ishift) = single_correlation_align(potential * reference, forcedalign)
        nd.shift(align[:,:,i],ishift,output = align[:,:,i])
        shift.append(ishift)                                  

# plot residue to reference
#        align[:,:,i] = reference - align[:,:,i] 

#        align[:,:,i] = reference
        

    return align, shift

def single_correlation_align(reference, align):
    'see crosscorrelation before stacks were cool'

    correlation = nd.correlate(align, reference, mode = 'constant')
    maxcorrelation = correlation.argmax()
    centerref = np.array([reference.shape[0]/2,reference.shape[1]/2])

    # print "maxcorrelation = %s" %maxcorrelation 

    # print "centerref = " 
    # print centerref 

    shift = centerref - np.array(np.unravel_index(maxcorrelation,correlation.shape)) 

    nd.shift(align,shift, output = align)

    return (align, shift)
#    return (correlation, shift)


def mask_align(align, threshold = 5, alignment = (0,0)):
    'masks the aling array by thresholding.\n Aligns to the alighnment = (1,1) = top - left/n(-1,1) = btm - left/n(1,-1) = top - right/n(-1,-1) = bottom - right corner. A value of 0 in alignment does not align that direction.\n Returns the align array aligned to the first array in the stack and a list of the shift (r[0]-a[0],r[1]-a[1]).'

    
    maskreference = np.where(align[:,:,0] > threshold, 1, 0)
    
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
        refrows   = maskreference.sum(0)[::alignment[1]]   
        i        = 0
        while not refrows[i]:        
            drefrows += 1
            i        += 1

    for k in range(1,align.shape[2]):
        kshift = [0,0]
        maskalign     = np.where(align[:,:,k] > threshold, 1, 0)
        if alignment[0]:
            aligncols = maskalign.sum(1)[::alignment[0]]
            dcols  = 0
            i = 0
            while not aligncols[i]:        
                dcols += 1
                i    += 1
            kshift[0] = dcols - drefcols
        if alignment[1]:
            alignrows = maskalign.sum(0)[::alignment[1]]
            drows  = 0
            i = 0
            while not alignrows[i]:        
                drows += 1
                i    += 1
                kshift[0] = drows - drefrows

        shift.append(kshift)
        align[:,:,k] = nd.shift(align[:,:,k],shift[k])

    return (align, shift)


# def sift_align(align, threshold = None):
#     'uses the silx.image.sift alignment. Returns aligned array and tupel  (r[0]-a[0],r[1]-a[1]). The optional threshold has not been shown to help for single wire images.'

# # initialize sift according to mail from Pierre Paleo

#     reference = np.ascontiguousarray(np.where(align[:,:,0] > threshold, align[:,:,0], 0))

#     alignplan = sift.LinearAlign(reference, devicetype="GPU")
# # alternative:
# #    alignplan = sift.LinearAlign(reference, device=(2,0))
    
#     shift = [(0,0)]
#     for i in range(1,align.shape[2]):
        
#         dummy = np.copy(align[:,:,i])
#         dummy = np.where(dummy > threshold, dummy, 0)

#         result  = alignplan.align(dummy, shift_only = True, return_all=True)
#         if result:
#             shift.append(np.round(np.asarray(result['offset'],dtype = float)))
#         else:
#             shift.append(np.zeros(shape=2))

#         align[:,:,i] = nd.shift(align[:,:,i],shift[i])



#     return (align, shift)

def shift_image(data, shift):
    for i, ishift in enumerate(shift):
        nd.shift(data[:,:,i], ishift, output = data[:,:,i],)
    
    return data

def real_from_rel(frame,data,shift = [1,1]):
    'this function gives you the real frame number if you give it the shifted frame number, an examplary dataset and the shift.\n returns -1 if the frame is out of the FOV.\n accepts slices'

    shift[0] = -shift[0]
    shift[1] = -shift[1]
    
    frames = np.arange(data.size)
    print frames
#    print shift
    frames = frames.reshape(data.shape)
    print frames
    frames = nd.shift(frames,shift,cval = -1)
    print frames
    frames = frames.flatten()
    print frames[frame]
    
    return frames[frame]

def image_align(align, mode = {'mode':'sift'}):
    'returns the align array stack aligned to the [0] array in it. The relative shift ist areturned as a list of touples (r[0]-a[0],r[1]-a[1])'

    # if mode['mode'] == 'sift':
    #     (align, shift) = sift_align(align, threshold = mode['threshold'])
    if mode['mode'] == 'mask':
        (align, shift) = mask_align(align, threshold = mode['threshold'], alignment = mode['alignment'])
    elif mode['mode'] == 'crosscorrelation':
        (align, shift) = crosscorrelation_align(align, alignment = mode['alignment'])

    else:
        print "%s is not a valid mode" % mode    
        
    return (align, shift)




if __name__ == '__main__':
    'testfuction'

# setup an pair of array to test alignment 
    x         = np.atleast_2d(np.arange(100))
    y         = np.atleast_2d(np.arange(100)).T
    shift     = (5,15)
    reference = ((x-50)/100)**2 * ((y-50)/100)**2
    reference[50:55,50:55] = 50000
    align1    = nd.shift(reference,shift)
    align2    = nd.shift(align1,shift)
    align     = np.dstack((reference, align1, align2))
    dummy     = np.copy(align)

#    timing:
    start_time = timeit.default_timer()

# do the test    
    # (dummy, (foundshift)) = image_align(dummy, mode = {'mode':'sift'})
    # print 'sift shift found:'
    # print foundshift
    # print 'took %s' % (timeit.default_timer() - start_time)
    # dummy     = np.copy(align)
#    timing
    start_time = timeit.default_timer()
    (dummy, (foundshift)) = image_align(dummy, mode = {'mode':'mask','threshold':np.percentile(align,70),'alignment':(1,1)})
    print 'mask shift found:'
    print foundshift
    print 'took %s' % (timeit.default_timer() - start_time)
    dummy     = np.copy(align)
#    timing:
    start_time = timeit.default_timer()
    (dummy, (foundshift)) = image_align(dummy, mode = {'mode':'crosscorrelation'})
    print 'crosscorrelation shift found:'
    print foundshift
    print 'took %s' % (timeit.default_timer() - start_time)
    
    
    
