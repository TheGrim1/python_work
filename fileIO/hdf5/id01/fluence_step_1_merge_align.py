import h5py
import sys, os
import numpy as np
import time
import glob
import datetime
from shutil import rmtree


from silx.io.spech5 import SpecH5 as spech5

sys.path.append(os.path.abspath("/data/id13/inhouse2/AJ/skript"))
import simplecalc.image_align_elastix as ia
import simplecalc.image_deglitch as idg
import fileIO.images.image_tools as it


def normalize_scans(imagestack):
    'preserves dtype, careful with rounding and neg values!' 
    norm = np.copy(imagestack)
    norm = it.normalize_to_border(norm)
    norm2 = np.copy(norm)
    
    for i, image in enumerate(norm2):
        norm[i] = image/np.percentile(image,99)
    return norm

# def get_ref(working_dir, alignment_counter):
#     source_spec_fname = glob.glob(working_dir+'/spec/*_fast_*')[0]
#     with spech5(source_spec_fname) as spec_f:

#         first_scan = spec_f.values()[0]
#         title = str(first_scan['title'].value).split()
#         map_shape = [int(title[8]),int(title[4])]
        
#     return ref_array

def do_align(working_dir, alignment_counter):
    dest_path = working_dir + '/aligned/'
    dest_fname = dest_path + working_dir.split(os.path.sep)[-1] + '{}_aligned.h5'.format(alignment_counter)

    if os.path.exists(dest_path):
        pass
        # rmtree(dest_path)
        # print('removed {}'.format(dest_path))

    else:
        os.mkdir(dest_path)

    source_spec_fname = glob.glob(working_dir+'/spec/*_fast_*')[0]
    
    with spech5(source_spec_fname) as spec_f:
        print('reading {}'.format(source_spec_fname))
        first_scan = spec_f.values()[0]
        title = str(first_scan['title'].value).split()
        map_shape = [int(title[8]),int(title[4])]
        y_range = abs(float(title[7]) - float(title[6]))
        x_range = abs(float(title[3]) - float(title[2]))
        
        temp = first_scan['instrument/positioners/eur1_sp'].value/10.          
        
        scan_list = list(spec_f.items())
        scan_list.sort()
        no_scans = len(scan_list)
        
        ori_data = np.zeros(tuple([no_scans]+map_shape),dtype=np.float32)
        del_list = np.zeros(no_scans,dtype=np.float32)
        eta_list = np.zeros(no_scans,dtype=np.float32)
        phi_list = np.zeros(no_scans,dtype=np.float32)
        
        with h5py.File(dest_fname, 'w') as dest_h5:
            print('writing to {}'.format(dest_fname))            
            for i,[scan_name,scan_g] in enumerate(scan_list):
                ori_data[i] = np.asarray(scan_g['measurement/{}'.format(alignment_counter)]).reshape(map_shape)
                del_list[i] = scan_g['instrument/positioners/del'].value
                eta_list[i] = scan_g['instrument/positioners/eta'].value
                phi_list[i] = scan_g['instrument/positioners/phi'].value

            merged_g= dest_h5.create_group('merged_data')
            counter_g = merged_g.create_group(alignment_counter)
            # alignment parameters
            fixed_image = int(no_scans/2)
            resolutions =  ['4','2','1']
            
            ori_g = counter_g.create_group('{}_original'.format(alignment_counter))
            ori_g.create_dataset(name=alignment_counter,data=ori_data)

            norm_data = normalize_scans(ori_data)
            norm_g = counter_g.create_group('{}_normalized'.format(alignment_counter))
            norm_g.create_dataset(name='{}_norm'.format(alignment_counter),data=norm_data)

            aligned_data, shift = ia.elastix_align(norm_data, mode
                                                   ='translation', fixed_image_no=fixed_image,
                                                   NumberOfResolutions = resolutions)
            
            align_g = counter_g.create_group('{}_aligned'.format(alignment_counter))
            align_g.create_dataset(name='{}_aligned'.format(alignment_counter), data=aligned_data)


            alignment = merged_g.create_group('alignment')
            alignment.attrs['NXprocess'] = 'NXprocess'
            alignment.create_dataset(name='shift',data=shift)
            alignment_parameters = alignment.create_group('alignment_parameters')
            alignment_parameters.attrs['script'] = ia.__file__
            alignment_parameters.attrs['function'] = 'elastix_align'
            alignment_parameters.attrs['signal'] = alignment_counter
            alignment_parameters.attrs['mode'] = 'translation'
            alignment_parameters.attrs['fixed_image_no'] = fixed_image
            alignment_parameters.attrs['NumberOfResolutions'] = resolutions

            
            axes_g = merged_g.create_group('axes')
            axes_g.create_dataset('phi',data=phi_list)
            axes_g.create_dataset('eta',data=eta_list)
            axes_g.create_dataset('del',data=del_list)
            axes_g.create_dataset('temp',data=temp)
            axes_g.create_dataset('x',data=np.linspace(0,x_range,map_shape[1]))
            axes_g.create_dataset('y',data=np.linspace(0,y_range,map_shape[0]))
            
            print('written to {}'.format(dest_fname))            
            
    return dest_fname
            


if __name__=='__main__':


    working_dir_list = glob.glob('/data/id13/inhouse2/AJ/data/ma3576/id01/analysis/fluence/KMAPS/KMAP_*')
    alignment_counter = 'roi2'

    for working_dir in working_dir_list:
        
        dest_fname = do_align(working_dir=working_dir,
                              alignment_counter=alignment_counter)
