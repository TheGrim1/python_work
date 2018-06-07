import sys, os
import h5py
import numpy as np
import PIL.Image as Image

# change these to suit you:
NO_PROCESSES = 1
if NO_PROCESSES >1:
    from multiprocessing import Pool
SAVE_PATH = '/data/id13/inhouse10/THEDATA_I10_1/d_2018-05-03_user_ch5315_mannsfeld/PROCESS/SESSION_sum/test/'


##### helper functions
def save_tif(data,savename,verbose=False):
    """
    gets a 2D numpy array of shape 
    (width, height)
    and saves it into imagefname 
    """
    img = Image.fromarray(data)
    if verbose:
        print("saving {}".format(os.path.realpath(savename)))
    img.save(savename)

def parse_master_fname(data_fname):
    master_path = os.path.dirname(data_fname)
    master_fname = os.path.basename(data_fname)[:os.path.basename(data_fname).find("data")]+'master.h5'
    return master_path + os.path.sep + master_fname


##### this does the work:
def sum_and_max_scan(args):

    master_fname = args[0]
    save_fname =args[1]
    verbose = args[2]
    
    if not os.path.exists(master_fname):
        raise ValueError('file not found '.format(master_fname))
        
    # open the h5 file:
    with h5py.File(master_fname) as master_h5:
        data_group = master_h5['entry/data']
        data_sum = np.zeros(shape = data_group['data_000001'].shape[1:3], dtype=np.uint32)
        data_max = np.zeros(shape = data_group['data_000001'].shape[1:3], dtype=np.int16)

        no_frames = 0
        #loop over the datasets in the h5 file:
        for key in data_group.keys():
            if verbose:
                print('reading {}'.format(key))
            # loop over the frames in each dataset:
            for i,frame in enumerate(data_group[key]):
                if verbose:
                    print('reading frame {}'.format(i))
                # sum:
                data_sum += frame
                no_frames += 1
                # max-proj:
                data_max = np.where(frame>data_max,frame,data_max)

    # thresholding away hot pixels:
    data_sum = np.where(data_sum>1e8,0,data_sum)
    data_max = np.where(data_max>1e8,0,data_max)

    data_avg = np.asarray(data_sum,dtype=np.float32)/no_frames
    data_max = np.asarray(data_max,dtype=np.int16)
    save_tif(data_avg, save_fname.format('avg'),verbose)
    save_tif(data_max, save_fname.format('max'),verbose)


##### this organises the work:
def main(args):
    
    master_fnames = [parse_master_fname(x) for x in args if x.find('.h5')]
    verbose = True
    todo_list = []
    for master_fname in master_fnames:

        save_fname = SAVE_PATH + os.path.basename(master_fname)[0:os.path.basename(master_fname).find('_master')] +  '_{}.tif'
        todo=[]
        todo.append(master_fname)
        todo.append(save_fname)
        todo.append(verbose)
        todo_list.append(todo)


    if NO_PROCESSES >1:
        pool = Pool(processes=NO_PROCESSSES)
        pool.map_async(sum_and_max_scan,todo_list)
        pool.close()
        pool.join()
    else:
        for todo in todo_list:
            sum_and_max_scan(todo)

if __name__=='__main__':
    usage =""" \n1) python <thisfile.py> <arg1> <arg2> etc.  \n2)
python <thisfile.py> -f <file containing args as lines> \n3) find
<*yoursearch* -> arg1 etc.> | xargs python <thisfile.py> """

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
    
    main(args)
