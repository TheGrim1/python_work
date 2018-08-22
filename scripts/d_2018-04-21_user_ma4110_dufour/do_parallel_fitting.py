
from multiprocessing import Pool
import os
import fitter
import subprocess

def main():
    noprocesses = 50
    find_tpl = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-21_user_ma4110_dufour/PROCESS/SESSION_INTEGRATE/all/integrated**.h5'  
    min_size = '250M'
    
    data_fname_list = subprocess.check_output('find {} -size +{}'.format(find_tpl,min_size),shell=True).split('\n')[:-1]
    dest_path = '/hz/data/id13/inhouse10/THEDATA_I10_1/d_2018-04-21_user_ma4110_dufour/PROCESS/SESSION_INTEGRATE/fitted/'

    dest_fname_list = [dest_path + 'fitted_' + '_'.join(os.path.basename(fname).split('_')[1:]) for fname in data_fname_list]
    print(len(data_fname_list))
    print(len(dest_fname_list))
    todo_list = []
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    print('starting {} processes to integrate'.format(noprocesses))
    
    for data_fname, dest_fname in zip(data_fname_list,dest_fname_list):
        print(data_fname)
        todo_list.append([data_fname, dest_fname, False])


    # #DEBUG, do only one:
    # fitter.do_fit(todo_list[0])
        
    print(todo_list)
    pool = Pool(processes=noprocesses)
    pool.map_async(fitter.do_fit,todo_list)
    pool.close()
    pool.join()


if __name__=='__main__':
    main()
