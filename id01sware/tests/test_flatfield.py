"""# generate the raw data files
h5.create_hdf5(    specfile = 'psic_nano_20150120.spec', \
                image_prefix = 'ff_mpx4', \
                specdir = '/mntdirect/_data_id01_inhouse/leake/projects/2015/commissioning/',\
                imagedir ='/mntdirect/_data_id01_inhouse/Jan/2015_01_comm/flatfield/',\
                scan_nos = [7,8,9,10,11,12,13,14,15,16,17,18,21],\
                output_fn='raw_data_')
                
                
"""

from id01lib.flatfield import Flatfield
import numpy as np

"""
#create a flatfield
ff_path = '/mntdirect/_data_id01_inhouse/leake/projects/2015/commissioning/'
scan_no = 7
h5fn = 'raw_data_%i.h5'%scan_no
ff_data0 = h5.get_scan_images(h5fn,scan_no)[:]
scan_no = 8
h5fn = 'raw_data_%i.h5'%scan_no
ff_data1 = h5.get_scan_images(h5fn,scan_no)[:]
scan_no = 9
h5fn = 'raw_data_%i.h5'%scan_no
ff_data2 = h5.get_scan_images(h5fn,scan_no)[:]

ff_init = Flatfield(np.c_[ff_data0[:,:,1:95],ff_data1[:,:,1:80]], ff_path,auto = True)#,ff_data2[:,:,1:95]
ff_init.calc_ff()
ff_init.dump2hdf5()
ff,ff_unc = ff_init.get_ff()

a,b = h5.read_ff_h5(ff_init.ff_path,fn='ff.h5')
h5.print_hdf5_file_structure(ff_init.ff_path+'ff.h5')


"""