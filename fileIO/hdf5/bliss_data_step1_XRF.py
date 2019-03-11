import h5py
import sys, os
import numpy as np
import time
import glob
import datetime
from shutil import rmtree

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import simplecalc.image_align as ia
import simplecalc.image_deglitch as idg

def init_h5_file(dest_path, saving_name, verbose =False):

    dest_fname = os.path.realpath(dest_path + saving_name + '_merged.h5')
    
    if os.path.exists(dest_fname):
        os.remove(dest_fname)
        print('removing {}'.format(dest_fname))

    print('\nwriting to file')
    print(dest_fname)
    return dest_fname

def do_fluo_merge(dest_fname, source_fname, verbose=False):

    print('reading fluorescence from counter file {} '.format(source_fname))
    
    with h5py.File(dest_fname,'w') as dest_h5:
        with h5py.File(source_fname,'r') as source_h5:
            phi_h5path = 'axes/phi'

            print(source_h5.items())
            print(dest_h5.items())
            
            phi_list = [[data_g[phi_h5path].value, data_g] for _, data_g in source_h5.items()]
            phi_list.sort()
            print(phi_list)
            data_g_list = [x[1] for x in phi_list]
            merged_data = dest_h5.create_group('merged_data')
            fluo_merged = merged_data.create_group('fluorescence')

            # setup groups in dest_h5
            map_shape = data_g['XRF'].shape
            x_pts = map_shape[1]
            y_pts = map_shape[0]
            phi_pts = len(data_g_list)


            axes = dest_h5['merged_data'].create_group('axes')
            axes.attrs['NX_class'] = 'NXcollection'
            axes.create_dataset('phi', dtype=np.float32, shape = (phi_pts,))
            kappa = source_h5.values()[0]['axes/kappa']
            axes.create_dataset('kappa', data=kappa)
            axes.create_dataset('x' ,data=range(x_pts))
            axes.create_dataset('y', data=range(y_pts))

            fluo_ori = fluo_merged.create_group('fluo_original')
            fluo_ori.attrs['NX_class'] = 'NXdata'
            fluo_ori.attrs['signal'] = 'XRF'
            fluo_ori.attrs['axes'] = ['phi','y','x']
            fluo_ori['phi'] = axes['phi']
            fluo_ori['x'] = axes['x']
            fluo_ori['y'] = axes['y']
            fluo_ori.create_dataset(name='XRF', dtype=np.uint64, shape=(phi_pts, y_pts, x_pts), compression='lzf', shuffle=False)

            fluo_aligned = fluo_merged.create_group('fluo_aligned')
            fluo_aligned.attrs['NX_class'] = 'NXdata'
            fluo_aligned.attrs['signal'] = 'XRF'
            fluo_aligned.attrs['axes'] = ['phi','y','x']
            fluo_aligned['phi'] = axes['phi']
            fluo_aligned['x'] = axes['x']
            fluo_aligned['y'] = axes['y']


            for i,[phi_pos, data_g] in enumerate(phi_list):
                print('reading no {} of {}'.format(i+1,phi_pts))
                print('phi {}, group {}'.format(phi_pos,data_g.name))
                # convert to uint32 and *1000 to avoid floats from here on
                fluo_data = np.asarray(data_g['XRF'],dtype=np.uint64)*1000            

                fluo_ori['XRF'][i]=np.asarray((fluo_data),dtype=np.uint64)
                axes['phi'][i] = phi_pos

            dest_h5.flush()

            print('aligning')

            fluo_data=np.asarray(fluo_ori['XRF'])

            XRF_al = np.copy(fluo_data)
            ref_frame = np.zeros_like(XRF_al[0])
            ref_frame[:,ref_frame.shape[1]/2] = XRF_al.max()
            XRF_al = np.stack([ref_frame]+list(XRF_al))

            # remove first,glitchy point per line
            XRF_al[:,:,0] = 0

            XRF_al, lines_shift = idg.imagestack_correct_lines_edge(imagestack=XRF_al,reference_frame_no=0, edge='right')
            
            res = idg.imagestack_shift_lines(fluo_data,-np.asarray(lines_shift[1:]))
            
            aligned, shift = ia.com_and_mask_align(res, com_axis=1, mask_direction=-1, threshold=np.percentile(fluo_data,90))

            fluo_aligned.create_dataset(name='XRF',data=aligned, compression='lzf')
            aligned_sum = aligned.sum(axis=0)
            aligned_sum_mean = aligned_sum.mean()
            mask = np.where(aligned_sum<aligned_sum_mean,0,1)
            fluo_aligned.create_dataset(name='mask',data=mask, compression='lzf')
            
            
            alignment = merged_data.create_group('alignment')
            alignment.attrs['NXprocess'] = 'NXprocess'
            alignment.create_dataset(name='shift',data=shift)
            alignment.create_dataset(name='lines_shift',data=np.asarray(lines_shift[1:]))
            alignment_parameters = alignment.create_group('alignment_parameters')
            alignment_parameters.attrs['script'] = ia.__file__
            alignment_parameters.attrs['function'] = 'com_and_mask_align'
            alignment_parameters.attrs['signal'] = fluo_ori.name
            alignment_parameters.attrs['com_axis'] = 1
            alignment_parameters.attrs['mask_direction'] = 1
            alignment_parameters.attrs['threshold'] = fluo_data.mean()
            alignment_parameters.attrs['alignment_order'] = 'inverted'
            
            dest_h5.flush()
        print('written to {}'.format(dest_fname))             

    
    
def main(preview_fname, saving_name, dest_path, troi_dict):
    verbose = True
    dest_fname =  init_h5_file(dest_path, saving_name, verbose=verbose)
    
    do_fluo_merge(dest_fname, source_fname=preview_fname, verbose=verbose)


if __name__ == '__main__':
        
    # session_name = 'alignment'
    # saving_name = 'kmap_rocking'
    # map_shape = (140,80)
    # troi_dict = {'red':np.asarray([[2105,645],[30,30]]),
    #              'blue':np.asarray([[1262,1780],[1284-1262,1800-1780]])}

    # session_name = 'day_two'
    # saving_name = 'kmap_and_cen_4b'
    # troi_dict = {'red':np.asarray([[995,210],[1018-995,235-210]]),
    #              'blue':np.asarray([[497,1192],[513-497,1232-1192]]),
    #     	 'green':np.asarray([[760,1800],[800-760,1840-1800]])}

    # session_name = 'day_two'
    # saving_name = 'kmap_and_cen_3b'
    # troi_dict = {'black':np.asarray([[1306,600],[1327-1306,636-600]]),
    #              'yellow':np.asarray([[1505,1404],[1523-1505,1422-1404]]),
    #     	 'cyan':np.asarray([[392,1685],[409-392, 1702-1685]])}

    session_path = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/DATA/'+session_name+ '/eh3/'

    dest_path = '/data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/PROCESS/previews/'+session_name +'/'
    
    preview_file = dest_path +'/'+ saving_name + '/'+ saving_name + '_preview.h5'
    
    main(preview_file, saving_name, dest_path, troi_dict)
