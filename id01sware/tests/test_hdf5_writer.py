from id01lib.hdf5_writer import *

if __name__ == "__main__":
    #from hdf5_writer import create_hdf5
    create_hdf5(specfile = 'Pt.spec', \
                specdir = '/mntdirect/_data_id01_inhouse/leake/projects/2014/bragg-CDI/20141204-pt',\
                imagedir ='/mntdirect/_data_id01_inhouse/hs4670/Pt/',\
                scan_nos = [8],\
                output_fn='raw_data_',\
                savetiff = 'testing',\
                tiffdir='tifftest/')
