import numpy as np
from scipy.ndimage import shift as ndshift
from scipy.ndimage.filters import gaussian_filter as gauss_fil
import fabio
import h5py

import sys,os
sys.path.append('/data/id13/inhouse2/AJ/skript/')
import simplecalc.fitting2d as fit2d


def shift_freqaxis(array, copy=True):
    '''
    corrected version to move freq = 0 to the middle of the array
    only 2D
    with ndshift mode='wrap' the last line is lost 
    '''
    if copy:
        data = np.copy(array)
    else:
        data = array

    return np.fft.fftshift(data,axes=tuple(np.arange(array.ndim-1)))
   
def shiftback_freqaxis(array, copy=True):
    '''
    corrected version to move freq = 0 to the middle of the array
    only 2D
    with ndshift mode='wrap' the last line is lost 
    '''
    if copy:
        data = np.copy(array)
    else:
        data = array

    return np.fft.ifftshift(data,axes=tuple(np.arange(array.ndim-1)))

def get_power(array,shift = True):
    power_spec = np.asarray(np.abs(array)**2,dtype=np.uint64)
    if shift:
        return shift_freqaxis(power_spec)
    else:
        return power_spec

def get_amplitude(array, shift=True):
    amp_spec = np.asarray(np.abs(array),dtype=np.uint64)
    if shift:
        return shift_freqaxis(amp_spec)
    else:
        return np.sqrt(amp_spec)
    

def get_phase(array,shift=True,deg=True,power_spec=None):
    '''
    shifted only for 2D arrrays
    '''
    if type(power_spec)==type(None):
        power_spec = get_power(array,shift=False)
            
    threshold = power_spec.max()/1000
    phase_spec = np.where(power_spec>threshold,np.angle(array,deg=deg).astype(np.float32),0)

    if shift:
        return shift_freqaxis(phase_spec)
    else:
        return phase_spec

def get_carrier_freq(array, mode='max'):
    '''
    returns the main, 'carrier' frequency of the signal array
    modes == 'avg', 'gauss_fit', 'max'
    gauss_fit is 2-dim only
    '''

    power_spec = get_power(array,shift=True)
    power_spec -= power_spec.mean()
    power_spec = np.where(power_spec>0,power_spec,0)
    
    ranges = [np.linspace(-(x-1)/2,(x-1)/2,x) for x in array.shape[:-1]]
    ranges.append(np.linspace(0,array.shape[-1]-1,array.shape[-1]))

    if mode.upper()== 'MAX':
        carrier_index = np.unravel_index(np.argmax(power_spec, axis=None), power_spec.shape)
        return [ranges[x][ind] for x,ind in enumerate(carrier_index)]

    elif mode.upper()== 'AVG':
        meshgrid = np.asarray(np.meshgrid(*ranges,indexing='ij'))
        return np.asarray([np.average(x,weights=power_spec) for x in meshgrid])    

    elif mode.upper()== 'GAUSS_FIT':
        assert array.ndim == 2
        meshgrid = np.asarray(np.meshgrid(*ranges,indexing='ij'))
        fit_result = fit2d.do_gauss2d_fit(power_spec,x=meshgrid[1],y=meshgrid[0],force_positive=True)
        return [fit_result[0][0],fit_result[0][1]]
    else:
        raise NotImplementedError('mode {} is not implemented'.format(mode))
    

def get_carrier_phase(array,mode='max'):
    '''
    returns the phase at the 'carrier' frequency of the signal array (2-dim, shift_corrected)
    modes == 'avg', 'gauss_fit', 'max'
    '''

    power_spec = get_power(array,shift=True)
    phase_spec = get_phase(array, shift=True)
    power_spec -= power_spec.mean()
    power_spec = np.where(power_spec>0,power_spec,0)
    
    ranges = [np.linspace(-(x-1)/2,(x-1)/2,x) for x in array.shape[:-1]]
    ranges.append(np.arange(array.shape[-1]))

    if mode.upper()== 'MAX':
        carrier_index = np.unravel_index(np.argmax(power_spec, axis=None), power_spec.shape)
        return phase_spec[carrier_index]

    elif mode.upper()== 'AVG':
        
        return np.average(phase_spec,weights=power_spec)    

    elif mode.upper()== 'GAUSS_FIT':

        meshgrid = np.asarray(np.meshgrid(*ranges,indexing='ij'))
        fit_result = fit2d.do_gauss2d_fit(power_spec,x=meshgrid[1],y=meshgrid[0],force_positive=True)
        gauss_weight = fit2d.gauss2d_func(fit_result[0], x=meshgrid[1],y=meshgrid[0],force_positive=True)
        return np.average(phase_spec,weights=gauss_weight)
    
    else:
        raise NotImplementedError('mode {} is not implemented'.format(mode))

def get_background(edf_fname_list):
    '''
    simple avg of all frames. Assumes sufficiently heterogenous (esp. phase!) scan.
    '''

    sum_frame = np.zeros(shape=fabio.open(edf_fname_list[0]).dims,dtype=np.int64)
    for fname in edf_fname_list:
        sum_frame += fabio.open(fname).data

    mean_frame = np.asarray(sum_frame/len(edf_fname_list),dtype = np.uint16)
    
    return mean_frame

def save_interferometer_scan(source_path='/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/DATA/test_scans/eh3/',
                             save_path='/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/',
                             scanname='dmesh_1',
                             mapshape = None,
                             background_fname='/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/bkg_frame.edf'):

    path = source_path + os.path.sep + scanname
    edf_fname_list = [path+os.path.sep+x for x in os.listdir(path) if x.find('.edf')>0]
    edf_fname_list.sort()
    
    path = os.path.realpath(save_path+os.path.sep+scanname+os.path.sep)
    if not os.path.exists(save_path+os.path.sep+scanname):
        os.makedirs(path)

    if scanname.find('kmap')>=0:
        is_kmap = True
        kmap_edf = fabio.open(edf_fname_list[0])
        no_frames = kmap_edf.nframes  
    else:
        is_kmap = False
        no_frames= len(edf_fname_list)
        
    if type(mapshape)==type(None):
        mapshape=tuple([no_frames])


    background_frame = fabio.open(background_fname).data
    real_shape = list(mapshape) + list(background_frame.shape)
    real_dtype = background_frame.dtype

    Background = np.fft.rfft2(background_frame)
    background_power = get_power(Background)
    background_amplitude = get_amplitude(Background)
    background_phase = get_phase(Background, power_spec = shiftback_freqaxis(background_power))
    fft_shape = list(mapshape) + list(background_power.shape)
    power_dtype = background_power.dtype
    amplitude_dtype = background_amplitude.dtype
    phase_dtype = background_phase.dtype

    indexes = np.meshgrid(*[range(x) for x in mapshape],indexing='ij')
    index_list = zip(*[x.flatten() for x in indexes])
    
    with h5py.File(path+os.path.sep+scanname+'.h5','w') as h5_f:
        entry=h5_f.create_group('entry')
        fft_group=entry.create_group('fft')
        real_group = entry.create_group('real')
        background_group = entry.create_group('background')

        background_group.create_dataset('real',
                                        data=background_frame,
                                        compression='lzf')
        background_group.create_dataset('power_spec',
                                        data=background_power,
                                        compression='lzf')
        background_group.create_dataset('amplitude_spec',
                                        data=background_amplitude,
                                        compression='lzf')
        background_group.create_dataset('phase_spec',
                                        data=background_phase,
                                        compression='lzf')

        
        real_ds=real_group.create_dataset('frames',shape=real_shape,dtype=real_dtype,compression='lzf',chunks=tuple([1]*len(mapshape)+[real_shape[-2],real_shape[-1]]))
        power_ds=fft_group.create_dataset('power_spec',shape=fft_shape,dtype=power_dtype,compression='lzf',chunks=tuple([1]*len(mapshape)+[fft_shape[-2],fft_shape[-1]]))
        amplitude_ds=fft_group.create_dataset('amplitude_spec',shape=fft_shape,dtype=amplitude_dtype,compression='lzf',chunks=tuple([1]*len(mapshape)+[fft_shape[-2],fft_shape[-1]]))
        phase_ds=fft_group.create_dataset('phase_spec',shape=fft_shape,dtype=phase_dtype,compression='lzf',chunks=tuple([1]*len(mapshape)+[fft_shape[-2],fft_shape[-1]]))

      
        for i,index in enumerate(index_list):
            print('reading {} of {} frames'.format(i+1,no_frames))
            if is_kmap:
                kmap_edf.currentframe = i
                frame = kmap_edf.data
            else:

                fname = edf_fname_list[i]
                frame = fabio.open(fname).data

            # ## DEBUG
            # print(frame[100,100])
            Frame = np.fft.rfft2(frame)
            Frame = Frame - Frame[0,0]/Background[0,0]*Background
            real_ds[index] = frame
            power_spec = get_power(Frame)
            amplitude_spec = get_amplitude(Frame)
            power_ds[index] = power_spec
            amplitude_ds[index] = amplitude_spec
            phase_ds[index] = get_phase(Frame, power_spec=shiftback_freqaxis(power_spec))           

def extract_carrier(frame, carrier_freq_vector):
    '''
    extract a normalised intensity profile along the axis of the carrier frequency vector
    frame = 2d real image
    returns x [pxl] y [cnts]
    0 is in the middle of x
    '''
    ratio = carrier_freq_vector[0]/carrier_freq_vector[1]

    out_ratio=[ratio,1/ratio]

    diag = int(np.sqrt(frame.shape[0]**2+frame.shape[1]**2)+1)
   
    if ratio>1:
        ratio = 1/ratio
    elif ratio<-1:
        frame = frame[:-1]
        ratio = 1/ratio
    elif ratio <0:
        frame = frame.swapaxis(0,1)
        frame = frame[:-1]
    else:
        frame = frame.swapaxis(0,1)      

    shifted_frame = np.pad(frame,[[0,diag-frams.shape[0]],[0,diag-frame.shape[1]]],mode='constant')
    for i,line in enumerate(shifted_frame):
    shifted_frame[i]=ndshift(line,i*ratio,mode='constant',cval=0,order=1)

    
    mean_counts = np.average(shifted_frame,wheights=np.where(shifted_frame>0,1,0))
    mean_counts = np.asarray([x for x in mean_counts if x>0])
    

    step_lenth = np.sqrt(1+ratio**2)    
    x = np.linspace(-0.5*step_length,0.5*step_length,len(mean_counts))
                
    return shifted_frame.sum(1)

    

def sin_fit_carrier(frame, carrier_freq_vector):
    '''
    phase = 0 in the middle of the frame!
    '''
    return amplitude, period, phase, periods



            
    
def evaluate_inteferometer_scan(source_fname, derst_fname=None):
    amp_datapath = 'entry/fft/amplitude_spec'

    if dest_fname==None:
        dest_fname = source_fname[:,source_fname.find('.h5')]+'_eval.h5'
        
    with h5py.File(source_fname,'r') as source_h5:
        amp_spec_ds = source_h5[amp_datapath]
        real_ds = source_h5['entry/real/frames']
        data_shape = amp_spec_ds.shape
        mapshape = data_shape[:-2]
        frame_shape = data:shape[-2::]
        
        indexes = np.meshgrid(*[range(x) for x in mapshape],indexing='ij')
        index_list = zip(*[x.flatten() for x in indexes])
        no_frames= len(index_list)
    
        ranges = [np.linspace(-(x-1)/2,(x-1)/2,x) for x in array.shape[:-1]]
        ranges.append(np.arange(array.shape[-1]))
        meshgrid = np.asarray(np.meshgrid(*ranges,indexing='ij'))
        
        with h5py.File(dest_fname,'w') as dest_h5:
            entry=dest_fname.create_group('entry')
            fft_eval = entry.create_group('fft_eval')

            carrier_freq_vector_ds = fft_eval.create_dataset('carrier_frequecy_vector',shape = list(mapshape)+[2], dtype=np.float32)
            carrier_vector_ds.attrs['unit'] = '1/frame length'
            carrier_period_vector_ds = fft_eval.create_dataset('carrier_period_vector',shape = list(mapshape)+[2], dtype=np.float32)
            carrier_period_vector_ds.attrs['unit'] = 'pxl'
            
            amp_ds = fft_eval.create_dataset('amplitude',shape = list(mapshape), dtype=np.float32)
            amp_ds.attrs['unit'] = 'counts'
            period_ds = fft_eval.create_dataset('period',shape = list(mapshape), dtype=np.float32)
            period_ds.attrs['unit'] = 'pxl'
            phase_ds = fft_eval.create_dataset('phase',shape = list(mapshape), dtype=np.float32)
            phase_ds.attrs['unit'] = 'degree'
            phi_period_ds = fft_eval.create_dataset('phi_period',shape = list(mapshape), dtype=np.float32)
            phi_period_ds.attrs['unit'] = 'pxl'
            kappa_period_ds = fft_eval.create_dataset('kappa_period',shape = list(mapshape), dtype=np.float32)
            kappa_period_ds.attrs['unit'] = 'pxl'
            
            for i, index in enumerate(index_list):
                print('evaluating frame {} of {}'.format(i+1,no_frames))
                amp_spec = amp_spec_ds[index]
                real_frame = real_ds[index]
                fit_result = fit2d.do_gauss2d_fit(amp_spec,x=meshgrid[1],y=meshgrid[0],force_positive=True)

                carrier_freq_vector_ds[index]=np.asarray([fit_result[0][0],fit_result[0][1]])
                carrier_period_vector_ds[index] = np.asarray([real_frame.shape[0]/fit_result[0][0],real_frame[1]/fit_result[0][1]])

                sin_fit_result = sin_fit(amp_spec,np.asarray([fit_result[0][0],fit_result[0][1]]))

                amp_ds[index] = sin_fit_result[0]
                period_ds[index] = sin_fit_result[1]
                phase_ds[index] = sin_fit_result[2]
                kappa_period_ds[index] = sin_fit_result[3][0]
                phi_period_ds[index] = sin_fit_result[3][1]
                                

            
        
if __name__=='__main__':
    scanname = sys.argv[1]

    if len(sys.argv) == 4:
        mapshape = (int(sys.argv[2]), int(sys.argv[3]))
        print('mapshape = ')
        print(mapshape)
    else:
        mapshape=None
        
    save_interferometer_scan(scanname=scanname, mapshape=mapshape)

                                

