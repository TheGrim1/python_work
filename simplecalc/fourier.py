import numpy as np
from scipy.ndimage import shift as ndshift
from scipy.ndimage.filters import gaussian_filter as gauss_fil
import fabio
import h5py

import sys,os
sys.path.append('/data/id13/inhouse2/AJ/skript/')
import simplecalc.fitting2d as fit2d
from simplecalc.fitting import do_sin_fit


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

def get_carrier_period(array, mode='max', threshold=None):
    '''
    returns the main, 'carrier' frequency of the signal array
    modes == 'avg', 'gauss_fit', 'max'
    gauss_fit is 2-dim only
    refine with sin_fit_carrier
    '''

    array[0,0] *=0
    power_spec = get_power(array,shift=True)

    ranges = []
    lengths= []
    for dim in power_spec.shape[:-1]:
        lengths.append(dim)
        if dim%2==0:
            ranges.append(np.linspace(int(dim/2),-int(dim/2)+1,dim))
        else:
            ranges.append(np.linspace(int(dim/2),-int(dim/2),dim))
    ranges.append(np.linspace(0,power_spec.shape[-1],power_spec.shape[-1]))
    lengths.append(2*power_spec.shape[-1])

    if type(threshold)!= type(None):
        if power_spec.max() < threshold:
            return np.asarray([1]*power_spec.ndim)
    
    if mode.upper()== 'MAX':
        carrier_index = np.unravel_index(np.argmax(power_spec, axis=None), power_spec.shape)
        freq= [ranges[x][ind] for x,ind in enumerate(carrier_index)]

    elif mode.upper()== 'MAX_AVG':
        max_index = np.unravel_index(np.argmax(power_spec, axis=None), power_spec.shape)
        meshgrid = np.asarray(np.meshgrid(*ranges,indexing='ij'))
        weights = np.zeros_like(power_spec)
        for i in [-2,-1,0,1,2]:
            for j in [-2,-1,0,1,2]:
                weights[max_index[0]+i,max_index[1]+j]+=power_spec[max_index[0]+i,max_index[1]+j]
        freq= np.asarray([np.average(x,weights=weights) for x in meshgrid])    

    elif mode.upper()== 'GAUSS_FIT':
        assert array.ndim == 2
        meshgrid = np.asarray(np.meshgrid(*ranges,indexing='ij'))
        power_spec*=power_spec-int(np.ceil(power_spec.mean()))
        power_spec=np.where(power_spec<0,0,power_spec)
        fit_result = fit2d.do_gauss2d_fit(power_spec,x=meshgrid[1],y=meshgrid[0],force_positive=True)
        freq= [fit_result[0][1],fit_result[0][0]]
    else:
        raise NotImplementedError('mode {} is not implemented'.format(mode))
    return np.asarray([x/f for x,f in zip(lengths,freq)])

def get_carrier_phase(array,mode='max'):
    '''
    NOT RECOMENDED, do sin_fit_carrier
    returns the phase at the 'carrier' frequency of the signal array (2-dim, shift_corrected)
    modes == 'avg', 'gauss_fit', 'max'
    '''

    power_spec = get_power(array,shift=True)
    phase_spec = get_phase(array, shift=True)
    power_spec -= power_spec.mean()
    power_spec = np.where(power_spec>0,power_spec,0)

    ranges = []
    lengths= []
    for dim in power_spec.shape[:-1]:
        lengths.append(dim)
        if dim%2==0:
            ranges.append(np.linspace(int(dim/2),-int(dim/2)+1,dim))
        else:
            ranges.append(np.linspace(int(dim/2),-int(dim/2),dim))

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

    sum_frame = np.zeros(shape=fabio.open(edf_fname_list[0]).data.shape,dtype=np.int64)
    for fname in edf_fname_list:
        sum_frame += fabio.open(fname).data

    mean_frame = np.asarray(sum_frame/len(edf_fname_list),dtype = np.uint16)
    
    return mean_frame

def extract_carrier(frame, carrier_vector):
    '''
    extract a normalised intensity profile along the axis of the carrier (period) vector
    frame = 2d real image
    returns x [pxl] y [cnts]
    0 is in the middle of x
    '''
    ratio = float(carrier_vector[1])/carrier_vector[0]

    work_frame=np.copy(frame)
    
    if ratio>1:
        work_frame = work_frame[::-1]
        ratio = 1/ratio
        work_frame = work_frame.swapaxes(0,1)
        padded_frame = np.pad(work_frame,[[0,0],[0,work_frame.shape[0]]],mode='constant')
    elif ratio<-1:
        ratio = -1/ratio
        work_frame = work_frame.swapaxes(0,1)
        padded_frame = np.pad(work_frame,[[0,0],[0,work_frame.shape[0]]],mode='constant')
    elif ratio <0:
        ratio *= -1
        padded_frame = np.pad(work_frame,[[0,0],[0,work_frame.shape[0]]],mode='constant')
    else:
        work_frame = work_frame[::-1]
        padded_frame = np.pad(work_frame,[[0,0],[0,work_frame.shape[0]]],mode='constant')
    
    shifted_frame = np.zeros_like(padded_frame,dtype=np.float32)
    
    for i,line in enumerate(padded_frame):
        shifted_frame[i]=ndshift(line,i*ratio,mode='constant',cval=0,order=0)
    
    mean_counts = np.average(shifted_frame,weights=np.where(shifted_frame>0,1,0.00001),axis=0)
    mean_counts = np.asarray([x for x in mean_counts if x>0.001])
    

    step_length = 1./np.sqrt(1+ratio**2)
    steps=len(mean_counts)
    pxl = np.linspace(-0.5*step_length*steps,0.5*step_length*steps,steps)
                
    return np.asarray([pxl, mean_counts])

    

def sin_fit_carrier(frame, carrier_vector):
    '''
    phase = 0 in the middle of the frame!
    '''
    carrier = extract_carrier(frame, carrier_vector)

    a = float(carrier_vector[0])
    b = float(carrier_vector[1])
    c = np.sqrt(a**2+b**2)
    period_guess=np.abs(a*b/c) # height in the phase triangle
    res = do_sin_fit(carrier.T,verbose=False,period_guess=np.linspace(max(period_guess-15,2.),period_guess+15,90))

    amplitude = res[0]
    phase = res[1]
    period = res[2]
    
    periods = np.asarray([period/period_guess*a,
                         period/period_guess*b])
    
    return [amplitude, period, phase, periods]


def save_interferometer_scan(source_path='/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/DATA/test_scans/eh3/',
                             save_path='/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/',
                             scanname='dmesh_1',
                             mapshape = None,
                             background_fname='/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/bkg_frame.edf',
                             fft_mask_fname='/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/fft_mask.edf'):


    path = source_path + os.path.sep + scanname
    edf_fname_list = [path+os.path.sep+x for x in os.listdir(path) if x.find('.edf')>0]
    edf_fname_list.sort()
    
    path = os.path.realpath(save_path+os.path.sep+scanname+os.path.sep)
    if not os.path.exists(save_path+os.path.sep+scanname):
        os.makedirs(path)
    save_fname = path+os.path.sep+scanname+'_fft.h5'
    if scanname.find('kmap')>=0:
        is_kmap = True
        kmap_edf = fabio.open(edf_fname_list[0])
        no_frames = kmap_edf.nframes  
    else:
        is_kmap = False
        no_frames= len(edf_fname_list)
        
    if type(mapshape)==type(None):
        mapshape=tuple([no_frames])


    background_frame = gauss_fil(fabio.open(background_fname).data,3)
    real_shape = list(mapshape) + list(background_frame.shape)
    real_dtype = background_frame.dtype
    mask_neg = np.where(fabio.open(fft_mask_fname).data,0,1)
    Background = np.fft.rfft2(background_frame)
    fft_dtype = Background.dtype
    background_power = get_power(Background)
    background_amplitude = get_amplitude(Background)
    background_phase = get_phase(Background, power_spec = shiftback_freqaxis(background_power))
    fft_shape = list(mapshape) + list(background_power.shape)
    power_dtype = background_power.dtype
    amplitude_dtype = background_amplitude.dtype
    phase_dtype = background_phase.dtype

    indexes = np.meshgrid(*[range(x) for x in mapshape],indexing='ij')
    index_list = zip(*[x.flatten() for x in indexes])
    
    with h5py.File(save_fname,'w') as h5_f:
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
        background_group.create_dataset('fft',
                                        data=Background,
                                        compression='lzf')

        
        real_ds=real_group.create_dataset('frames',shape=real_shape,dtype=real_dtype,compression='lzf',chunks=tuple([1]*len(mapshape)+[real_shape[-2],real_shape[-1]]))
        power_ds=fft_group.create_dataset('power_spec',shape=fft_shape,dtype=power_dtype,compression='lzf',chunks=tuple([1]*len(mapshape)+[fft_shape[-2],fft_shape[-1]]))
        fft_ds=fft_group.create_dataset('fft',shape=fft_shape,dtype=fft_dtype,compression='lzf',chunks=tuple([1]*len(mapshape)+[fft_shape[-2],fft_shape[-1]]))
        amplitude_ds=fft_group.create_dataset('amplitude_spec',shape=fft_shape,dtype=amplitude_dtype,compression='lzf',chunks=tuple([1]*len(mapshape)+[fft_shape[-2],fft_shape[-1]]))
        phase_ds=fft_group.create_dataset('phase_spec',shape=fft_shape,dtype=phase_dtype,compression='lzf',chunks=tuple([1]*len(mapshape)+[fft_shape[-2],fft_shape[-1]]))

        pid=os.getpid()
        for i,index in enumerate(index_list):
            print('{} reading {} of {} frames'.format(pid,i+1,no_frames))
            if is_kmap:
                kmap_edf.currentframe = i
                frame = kmap_edf.data
            else:
                fname = edf_fname_list[i]
                frame = fabio.open(fname).data

            frame = gauss_fil(frame,3)
            
            Frame = np.fft.rfft2(frame)
            Frame = Frame - Frame[0,0]/Background[0,0]*Background
            Frame[0,:]*=0
            Frame[:,0]*=0
            Frame*=mask_neg
            fft_ds[index]=Frame
            real_ds[index] = frame
            power_spec = get_power(Frame)
            amplitude_spec = get_amplitude(Frame)
            power_ds[index] = power_spec
            amplitude_ds[index] = amplitude_spec
            phase_ds[index] = get_phase(Frame, power_spec=shiftback_freqaxis(power_spec))           

    return save_fname

def evaluate_inteferometer_scan(source_fname,
                                dest_fname=None):

    amp_datapath = 'entry/fft/amplitude_spec'

    if dest_fname==None:
        dest_fname = source_fname[:source_fname.find('.h5')]+'_eval.h5'
        
    with h5py.File(source_fname,'r') as source_h5:
        amp_spec_ds = source_h5[amp_datapath]
        real_ds = source_h5['entry/real/frames']
        fft_ds = source_h5['entry/fft/fft']
        data_shape = amp_spec_ds.shape
        mapshape = data_shape[:-2]
        frame_shape = data_shape[-2::]
        
        indexes = np.meshgrid(*[range(x) for x in mapshape],indexing='ij')
        index_list = zip(*[x.flatten() for x in indexes])
        no_frames= len(index_list)
    
        # ranges = [np.linspace(-(x-1)/2,(x-1)/2,x) for x in array.shape[:-1]]
        # ranges.append(np.arange(array.shape[-1]))
        # meshgrid = np.asarray(np.meshgrid(*ranges,indexing='ij'))
        
        with h5py.File(dest_fname,'w') as dest_h5:
            entry=dest_h5.create_group('entry')
            fft_eval = entry.create_group('fft_eval')

            # carrier_freq_vector_ds = fft_eval.create_dataset('carrier_frequecy_vector',shape = list(mapshape)+[2], dtype=np.float32)
            # carrier_vector_ds.attrs['unit'] = '1/frame length'
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
            pid = os.getpid()
            for i, index in enumerate(index_list):
                print('{} evaluating frame {} of {}'.format(pid,i+1,no_frames))

                real_frame = real_ds[index]
                Frame = fft_ds[index]
                
                carrier_period_vector = get_carrier_period(Frame, mode='max_avg', threshold = 100000)
                carrier_period_vector_ds[index] = carrier_period_vector

                sin_fit_result = sin_fit_carrier(real_frame,carrier_period_vector)

                amp_ds[index] = sin_fit_result[0]
                period_ds[index] = sin_fit_result[1]
                phase_ds[index] = sin_fit_result[2]
                kappa_period_ds[index] = carrier_period_vector[0]
                phi_period_ds[index] = carrier_period_vector[1]
                                
def do_full_process(args):
    source_path = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/DATA/test_scans/eh3/'
    
    save_path='/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/'

    scanname = args[0]
    mapshape = (100,100)
    background_fname='/data/id13/inhouse10/THEDATA_I10_1/d_2018-07-30_commi_BLISS_aj/PROCESS/FFT/bkg_frame.edf'
    
    print('{} doing {}'.format(os.getpid(),scanname))
    
    save_fname = save_interferometer_scan(source_path,
                                          save_path,
                                          scanname,
                                          mapshape,
                                          background_fname)
    
    evaluate_inteferometer_scan(source_fname=save_fname,
                                dest_fname=None)
            
        
if __name__=='__main__':
    scanname = sys.argv[1]
    
    if len(sys.argv) == 4:
        mapshape = (int(sys.argv[2]), int(sys.argv[3]))
        print('mapshape = ')
        print(mapshape)
    else:
        mapshape=None
        
    save_fname = save_interferometer_scan(scanname=scanname, mapshape=mapshape)

    evaluate_inteferometer_scan(source_fname=save_fname)

