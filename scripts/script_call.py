from subprocess import check_call


if __name__=='__main__':
    check_call('python /hz/data/id13/inhouse2/AJ/skript/fileIO/hdf5/step1_read_rois.py')
    check_call('python /hz/data/id13/inhouse2/AJ/skript/fileIO/hdf5/step2_merge.py')
