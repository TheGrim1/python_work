from __future__ import print_function
from __future__ import division
import numpy as np
import SimpleITK as sitk

def elastix_align(imagestack, mode = 'rigid', thetas= None, COR = None, fixed_image_no=0 , **parameterMapkwargs):
    '''
    uses elastix recognition to find the affine transform between imagesstack[0] and all others
    mode:
        'rigid'       : returns imagestack, shift, thetas
        'translation' : returns imagestack, shift

    default thetas == None
    else:
        + is more robust
        uses list of angeles thetas to first rotate imagestack[i] by -theta[i]
        rotates around COR or middle of images if COR == None
        then does alignment according to <rotation>
    the elastix parameterMap parameters maps can be passed in the dict parameterMapkwargs
    '''
    
    shift = []
    if type(thetas) == type(None):
        thetas = np.zeros(imagestack.shape[0])
    else:
        if COR == None:
            COR = ((imagestack.shape[1]/2.0),(imagestack.shape[2]/2.0))
        imagestack = rotate_series(imagestack, -np.asarray(thetas), COR = COR, copy = False)
    
    images = [sitk.GetImageFromArray(imagestack[i]) for i in range(imagestack.shape[0])]

    if mode == 'rigid':
        results = imagestack, shift, thetas
        parameterMap = sitk.GetDefaultParameterMap('rigid')
    elif mode == 'translation':
        results = imagestack, shift
        parameterMap = sitk.GetDefaultParameterMap('translation')
    else:
        raise ValueError('%s is not a valid mode argument' % mode)

    # default elastix parameters:
    parameterMap['MaximumNumberOfIterations'] = ['512']
    parameterMap['FinalBSplineInterpolationOrder'] = ['1']
    parameterMap['NumberOfResolutions'] = ['4','2']
    # custom parameters:
    for key, value in list(parameterMapkwargs.items()):
        parameterMap[key] = value

    fixedImage = images[fixed_image_no]
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.LogToConsoleOn()
    
    for i,image in enumerate(images):

        elastixImageFilter.SetMovingImage(image)
        elastixImageFilter.Execute()

        resultimage = elastixImageFilter.GetResultImage()
        imagestack[i] = sitk.GetArrayFromImage(resultimage)
        #imagestack[i] = np.where(sitk.GetArrayFromImage(resultimage)<0.1,0,sitk.GetArrayFromImage(resultimage))

        if mode == 'rigid':
            print('found parameters ', elastixImageFilter.GetTransformParameterMap()[0]['TransformParameters'])
            angle, dx, dy = elastixImageFilter.GetTransformParameterMap()[0]['TransformParameters']
            thetas[i] += (np.float(angle)/np.pi*180)
        elif mode == 'translation':
            print('found parameters ', elastixImageFilter.GetTransformParameterMap()[0]['TransformParameters'])
            dx, dy = elastixImageFilter.GetTransformParameterMap()[0]['TransformParameters']
            

        shift.append((np.float(dy),np.float(dx)))
        
    shift = np.asarray(shift)
    
    return results

