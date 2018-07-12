
import sys, os
import numpy as np
import fabio

from multiprocessing import Pool

sys.path.append('/data/id13/inhouse2/AJ/skript') 

from pythonmisc.worker_suicide import worker_init
from fileIO.edf.save_edf import save_edf

def _apply_mask_worker(args):
    
    [fname, mask, overwrite] = args

    data = fabio.open(fname).data
    data *= mask
    if overwrite:
        print('overwriting {}'.format(fname))
        save_edf(data, fname)
    else:
        save_fname = os.path.splitext(fname)[0]+'_masked.edf'
        print('overwriting {}'.format(fname))
        save_edf(data, fname)
        
    
    
def do_apply_mask_to_all(args):
    '''
    NOTE mask = 0 for values to mask!
    data_new = data_old*mask
    '''
    mask_fname = args.pop(0)
    edf_fname_list = [x for x in args if x.find('.edf')]

    todo_list = []
    overwrite=True

    mask = fabio.open(mask_fname).data
    
    for fname in edf_fname_list:
        todo=[fname, mask, overwrite]
        todo_list.append(todo)

    _apply_mask_worker(todo_list[0])

    pool = Pool(12,worker_init(os.getpid()))
    pool.map_async(_apply_mask_worker, todo_list)
    pool.close()
    pool.join() 
    
if __name__ == '__main__':
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc.  \n2)
python <thisfile.py> -f <file containing args as lines> \n3) find
<*yoursearch* -> arg1 etc.> | python <thisfile.py> """

    args = []
    if len(sys.argv) > 1:
        if sys.argv[1].find("-f")!= -1:
            f = open(sys.argv[2])
            for line in f:
                args.append(line.rstrip())
        else:
            args=sys.argv[1:]
    else:
        f = sys.stdin
        for line in f:
            args.append(line.rstrip())

            
    do_apply_mask_to_all(args)


