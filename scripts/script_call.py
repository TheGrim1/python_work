import os


def do_fsck():
    dest_list = ['/mntdirect/_data_id13_inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/DATA/day_two/eh3/kmap_and_cen_2/data.h5',
                 '/mntdirect/_data_id13_inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/DATA/day_two/eh3/kmap_and_cen_3/data.h5']

    source_list = ['/mntdirect/_data_id13_inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/DATA/day_two/eh3/kmap_and_cen_2/data_currupt.h5',
                   '/mntdirect/_data_id13_inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/DATA/day_two/eh3/kmap_and_cen_3/data_currupt.h5']

    for source, dest in zip(source_list, dest_list):
        call = (' ').join(['python','/data/id13/inhouse2/AJ/skript/scripts/fsck.py',source,dest])
        print(call)
        os.system(call) 

if __name__=='__main__':
    # os.system('python /hz/data/id13/inhouse2/AJ/skript/fileIO/hdf5/step1_read_rois.py')
    # os.system('python /hz/data/id13/inhouse2/AJ/skript/fileIO/hdf5/step2_merge.py')
    # os.system('python /hz/data/id13/inhouse2/AJ/skript/fileIO/hdf5/step3_Qxyz.py')
    # os.system('python /data/id13/inhouse2/AJ/skript/scripts/script_do_fft_analysis.py')
    # os.system('python /data/id13/inhouse2/AJ/skript/scripts/script_plot_fft_eval.py')
    # do_fsck()
    
    
