
import numpy as np
import h5py
import os, sys
import pyFAI
import scipy.ndimage.measurements as meas

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import pythonmisc.pickle_utils as pu
import simplecalc.lorentz_fitting as fit


def fit_data_employer(pickledargs_fname):
    '''
    gutwrenching way to (almost) completely uncouple the h5 access from the motherprocess
    Can be used to multiprocess.pool control the workers.
    '''
    fname =  __file__
    cmd = 'python {} {}'.format(fname, pickledargs_fname)
    
    os_response = os.system(cmd)
    if os_response >1:
        raise ValueError('in {}\nos.system() has responded with errorcode {} in process {}'.format(fname, os_response, os.getpid()))
    

def fit_data_worker(pickledargs_fname):
    '''
    interates fit routine over source_name[source_grouppath][dataset_path] and stores results in target_fname[target_group][results_path]
    these datasets will be created
    '''

    print('pid: {}'.format(os.getpid()))
    
    unpickled_args = pu.unpickle_from_file(pickledargs_fname, verbose = False)


    print(unpickled_args)
            
    source_fname = unpickled_args[0]
    source_grouppath = unpickled_args[1]
    dest_fname = unpickled_args[2]
    dest_grouppath = unpickled_args[3]
    no_lorentz = int(unpickled_args[4])
    lorentz_index_guess = unpickled_args[5]
    poly_degree =  int(unpickled_args[6])
    index_min, index_max= unpickled_args[7]
    verbose = unpickled_args[8]
        
    if verbose:
        print('='*25)
        print('process {} is doing'.format(os.getpid()))
        for arg in unpickled_args:
            print(arg)

    fit_parameter_names = []
    for x in range(no_lorentz):
        fit_parameter_names.append('maximum_{}'.format(x))
        fit_parameter_names.append('mean_{}'.format(x))
        fit_parameter_names.append('sigma_{}'.format(x))

    for i in range(poly_degree+1):
        fit_parameter_names.append('p_{}'.format(i))

    no_parameters = len(fit_parameter_names)
    no_datapoints = np.abs(index_max-index_min)
    
    with h5py.File(source_fname,'r') as source_h5:
        with h5py.File(dest_fname,'w') as dest_h5:
            entry = dest_h5.create_group('entry')
            entry.attrs['NXclass'] = 'NXentry'
            dest_group = dest_h5.create_group(dest_grouppath)
            source_group = source_h5[source_grouppath]
            data = source_group['I']
            q_range = source_group['q_radial'][index_min:index_max]
            frame_no = source_group['frame_no']
            datashape = data.shape
            no_frames = datashape[0]
            axes = dest_group.create_group('axes')

            axes.create_dataset(name='fit_parameter_number', data = np.arange(len(fit_parameter_names)))
            axes.create_dataset(name='fit_parameter_names', data = fit_parameter_names)
            axes.create_dataset(name='frame_no', data = frame_no)
            axes.create_dataset(name='q_range', data = q_range)
            axes['q_range'].attrs['units'] = 'nm^-1'

            fitted_parameters_ds = dest_group.create_dataset('fitted_parameters', dtype = np.float32, shape = (no_frames, no_parameters))
            original_data_ds = dest_group.create_dataset('original_data', dtype = np.float32, shape = (no_frames, no_datapoints),compression ='lzf')
            fitted_curve_ds = dest_group.create_dataset('fitted_curve', dtype = np.float32, shape = (no_frames, no_datapoints),compression ='lzf')
            residual_ds = dest_group.create_dataset('residual', dtype = np.float32, shape = (no_frames, no_datapoints),compression ='lzf')

            for i in range(no_frames):
                data_slice = data[i,index_min:index_max]
                fit_data = np.asarray(zip(q_range,data_slice))
                
                result = fit.do_multi_lorentz_and_poly_fit(fit_data,
                                                           no_lorentz,
                                                           poly_degree,
                                                           verbose,
                                                           lorentz_index_guess,
                                                           prefit=True)
                curve = fit.multi_lorentz_and_poly_func(result,
                                                        q_range,
                                                        no_lorentz)

                fitted_parameters_ds[i] = result
                original_data_ds[i] = data_slice
                fitted_curve_ds[i] = curve
                residual_ds[i] = data_slice - curve
                
            dest_group.attrs['NXclass'] = 'NXdata'
            dest_group['frame_no'] = axes['frame_no']
            dest_group['q_range'] = axes['q_range']
            dest_group.attrs['signal'] = 'fitted_curve'
            dest_group.attrs['axes'] = ['frame_no','q_range']
            

            dest_h5.flush()


            
    if verbose:
        print('process {} is done'.format(os.getpid()))
        print('='*25)


if __name__=='__main__':
    ''' 
    This is used by the local function integrate_troi_employer(pickledargs_fname),
    DO NOT CHANGE
    '''
    if len(sys.argv)!=2:
        print('usage : python integrate_troi_worker <pickled_instruction_list_fname>')
    pickledargs_fname = sys.argv[1]
    fit_data_worker(pickledargs_fname)
                        
