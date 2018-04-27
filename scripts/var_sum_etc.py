import numpy as np
import h5py
from silx import sx


def main():

    data1_fname='/hz/data/id13/inhouse7/THEDATA_I7_1/d_2017-05-20_inh_ihls2807_jc/DATA/AUTO-TRANSFER/eiger1/boaz_al2o3_calib_172_data_000001.h5'

    data2_fname='/hz/data/id13/inhouse7/THEDATA_I7_1/d_2017-05-20_inh_ihls2807_jc/DATA/AUTO-TRANSFER/eiger1/boaz_al2o3_calib_172_data_000002.h5'

    print('reading data1')
    with h5py.File(data1_fname) as hf_5:
        data1=np.asarray(h5_f['entry/data/data'])

    print('reading data2')
    with h5py.File(data2_fname) as hf_5:
        data2=np.asarray(h5_f['entry/data/data'])

    data=np.vstack([data1,data2])

    print('summing up')
    data_sum = data.sum(0)
    sx.imshow(data_sum)

    print('mask = np.where(data_sum>50000,1,0)')
    mask=np.where(data_sum>50000,1,0)
    print('data=np.where(mask,0,data)')
    data=np.where(mask,0,data)
    print('data_max=dat.max(0)')
    data_max=data.max(0)
    sx.imshow(data_max)

    print('data_avg=data_sum/data.shape[0]')
    data_avg=data_sum/(data.shape[0]*1.0)
    sx.imshow(data_avg)

    print('probs=[(np.where(data==i,1,0).sum(0)+np.where(data==i,1,0).sum(0)) for i in range(5)]')
    probs=[(np.where(data==i,1,0).sum(0)+np.where(data==i,1,0).sum(0)) for i in range(5)]

    for i in range(5):
        sx.imshow(probs[i])
    
    print('var=1/data.shape[0]*((data-data_avg)**2).sum()')
    var=1/data.shape[0]*((data-data_avg)**2).sum()
    
    sx.imshow(var)

    sx.imshow(var/data_avg)

    

if __name__=='__main__':
    main()
