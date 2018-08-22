
from multiprocessing import Pool
import os
import integrator
import subprocess

def main():
    noprocesses = 30

    find_tpl = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-21_user_ma4110_dufour/DATA/AUTO-TRANSFER/eiger1/**cycl**data**.h5'
    dest_path = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-21_user_ma4110_dufour/PROCESS/SESSION_INTEGRATE/all/'
    min_size = '250M'  

    # data_fname_list = ['/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-21_user_ma4110_dufour//DATA/AUTO-TRANSFER/eiger1/cell_cycleb3_65_1070_data_000001.h5', '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-21_user_ma4110_dufour/DATA/AUTO-TRANSFER/eiger1/cell_cycled0_31_1156_data_000001.h5','/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-21_user_ma4110_dufour/DATA/AUTO-TRANSFER/eiger1/cell_cycled0_117_1242_data_000001.h5']

    find_cmd = 'find {} -size +{}'.format(find_tpl,min_size)
    print(find_cmd)
    data_fname_list = subprocess.check_output(find_cmd, shell=True).split('\n')

    dest_fname_list = [dest_path + 'integrated_' + os.path.basename(fname) for fname in data_fname_list]
 

    print(len(dest_fname_list))
    tested_list = [fname for fname in dest_fname_list if not os.path.exists(fname)]
    print(len(tested_list))
    # don't do files that were already integrated
    data_fname_list = [x for x in data_fname_list if dest_path + 'integrated_' + os.path.basename(x) in tested_list]
    dest_fname_list = [dest_path + 'integrated_' + os.path.basename(fname) for fname in data_fname_list]

    
    
    
    todo_list = []
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    print('starting {} processes to integrate'.format(noprocesses))
    
    for data_fname, dest_fname in zip(data_fname_list,dest_fname_list):
        todo_list.append([data_fname, dest_fname, False])
        print(data_fname)

    # #DEBUG:
    # integrator.do_integration(todo_list[0])
        
    pool = Pool(processes=noprocesses)
    pool.map_async(integrator.do_integration,todo_list)
    pool.close()
    pool.join()

if __name__=='__main__':
    main()
