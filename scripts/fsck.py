

'''
(aj_venv)nanofocus:inhouse2/AJ/skript % pip install pyfive
DEPRECATION: Python 2.7 will reach the end of its life on January 1st, 2020. Please upgrade your Python as Python 2.7 won't be maintained after that date. A future version of pip will drop support for Python 2.7.
Requirement already satisfied: pyfive in /mntdirect/_data_id13_inhouse6/COMMON_DEVELOP/py_andreas/aj_venv/lib/python2.7/site-packages (0.3.0)
Requirement already satisfied: numpy in /mntdirect/_data_id13_inhouse6/COMMON_DEVELOP/py_andreas/aj_venv/lib/python2.7/site-packages (from pyfive) (1.16.0)
(aj_venv)nanofocus:inhouse2/AJ/skript % python scripts/fsck.py /data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/DATA//alignment/eh3/kmap_rocking2/data.h5 /data/id13/inhouse11/THEDATA_I11_1/d_2018-11-13_inh_ihma67_pre/DATA//alignment/eh3/kmap_rocking2/data_fixed.h5
Traceback (most recent call last):
  File "scripts/fsck.py", line 2, in <module>
    from pyfive import misc_low_level
ImportError: cannot import name misc_low_level
'''

#from pyfive import dataobjects
from pyfive import low_level
from pyfive import high_level
from pyfive.low_level import InvalidHDF5File
import h5py

def _copy_objects(name,src,dest,offset,level):
    obj = high_level.DataObjects(src,offset)
    if obj.is_dataset:
        try:
            data = obj.get_data()
        except:                 # reference,vlen_strings
            return
        dst_dataset = dest.create_dataset(name, obj.shape,
                                          dtype=obj.dtype,
                                          compression=obj.compression,
                                          data=data)
        return
    elif name == '/':
        child_dest = dest
    else:
        child_dest = dest.create_group(name)

    for name,offset in obj.get_links().items():
        try:
            _copy_objects(name,src,child_dest,offset,level+1)
        except InvalidHDF5File:
            if level:
                raise
            print("Entry corrupted: %s" % name)
            try:
                del dest[name]
            except KeyError:
                pass
            
def main(source_name,dest_name):
    with open(source_name,'rb') as src:
        sb = low_level.SuperBlock(src,0)
        with h5py.File(dest_name,'w') as dest:
            _copy_objects('/',src,dest,sb.offset_to_dataobjects,0)

if __name__ == '__main__':
    import sys
    main(sys.argv[1],sys.argv[2])
